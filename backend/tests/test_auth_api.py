"""Authentication routes.

These had no tests, which is how the OAuth callback shipped returning a JSON body: GitHub
navigates the *browser* to that endpoint, so the user would have landed on raw JSON on the API's
origin with no way back into the app.

Covered here: the capability flags the sign-in screen reads, the CSRF state check, the redirect
back to the application, token sign-in and the production block on it, and the session cookie's
flags.
"""

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.routes import auth as auth_routes
from app.config.settings import AppEnv, Settings
from app.models.core import GitHubConnection, User
from app.security.crypto import build_cipher

GITHUB_PROFILE = {
    "id": 5150,
    "login": "diksha",
    "name": "Diksha Kumari",
    "avatar_url": "https://example.test/avatar.png",
}


def github_transport(
    *,
    profile: dict | None = None,
    user_status: int = 200,
    capture: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)

        if request.url.path.endswith("/login/oauth/access_token"):
            return httpx.Response(200, json={"access_token": "gho_exchanged", "scope": "repo"})

        if request.url.path.endswith("/user"):
            return httpx.Response(user_status, json=profile or GITHUB_PROFILE)

        return httpx.Response(404, json={"message": "unexpected"})

    return httpx.MockTransport(handler)


@pytest.fixture
async def oauth_settings(settings: Settings) -> Settings:
    from cryptography.fernet import Fernet

    return settings.model_copy(
        update={
            "github_client_id": "client-id",
            "github_client_secret": "client-secret",  # noqa: S106 - test fixture
            "frontend_origin": "http://localhost:5173",
            # A fixed key, because the shared settings leave this unset and then every
            # build_cipher call invents a new ephemeral one that cannot read what the last
            # one wrote.
            "token_encryption_key": Fernet.generate_key().decode(),
        }
    )


async def client_with(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> AsyncClient:
    """An HTTP client bound to the app with the given settings."""
    from app.config.settings import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestCapabilityFlags:
    async def test_health_reports_which_sign_in_methods_work(
        self, client: AsyncClient
    ) -> None:
        """The sign-in screen reads these, so a button is never offered that would 503."""
        body = (await client.get("/health")).json()

        assert body["dev_login_available"] is True, "test env must allow token sign-in"
        assert body["github_oauth_configured"] is False, "no OAuth app in the test settings"

    async def test_oauth_configured_is_true_only_with_both_halves(
        self, session_factory, oauth_settings: Settings
    ) -> None:  # noqa: ANN001
        async with await client_with(session_factory, oauth_settings) as http:
            assert (await http.get("/health")).json()["github_oauth_configured"] is True

        half = oauth_settings.model_copy(update={"github_client_secret": None})

        async with await client_with(session_factory, half) as http:
            assert (await http.get("/health")).json()["github_oauth_configured"] is False

    async def test_production_hides_token_sign_in(
        self, session_factory, settings: Settings
    ) -> None:  # noqa: ANN001
        production = settings.model_copy(
            update={"app_env": AppEnv.production, "session_secret": "a-real-secret"}
        )

        async with await client_with(session_factory, production) as http:
            assert (await http.get("/health")).json()["dev_login_available"] is False


class TestOAuthStart:
    async def test_unconfigured_oauth_is_refused_rather_than_redirecting_nowhere(
        self, client: AsyncClient
    ) -> None:
        assert (await client.get("/auth/github/start")).status_code == 503

    async def test_authorize_url_carries_the_scopes_and_a_state(
        self, session_factory, oauth_settings: Settings
    ) -> None:  # noqa: ANN001
        async with await client_with(session_factory, oauth_settings) as http:
            url = (await http.get("/auth/github/start")).json()["authorize_url"]

        assert url.startswith("https://github.com/login/oauth/authorize?")
        assert "client_id=client-id" in url
        assert "scope=repo%20read:user" in url
        assert "state=" in url


class TestOAuthCallback:
    async def test_unknown_state_is_rejected(
        self, session_factory, oauth_settings: Settings
    ) -> None:  # noqa: ANN001
        """Without this, a link from anywhere could complete a sign-in."""
        async with await client_with(session_factory, oauth_settings) as http:
            response = await http.get(
                "/auth/github/callback", params={"code": "c", "state": "never-issued"}
            )

        assert response.status_code == 400

    async def test_state_cannot_be_replayed(
        self, session_factory, oauth_settings: Settings, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        async with await client_with(session_factory, oauth_settings) as http:
            state = _issued_state(
                (await http.get("/auth/github/start")).json()["authorize_url"]
            )

            first = await http.get(
                "/auth/github/callback",
                params={"code": "c", "state": state},
                follow_redirects=False,
            )
            second = await http.get(
                "/auth/github/callback",
                params={"code": "c", "state": state},
                follow_redirects=False,
            )

        assert first.status_code == 303
        assert second.status_code == 400, "a consumed state must not work twice"

    async def test_success_redirects_to_the_app_and_sets_the_session(
        self, session_factory, oauth_settings: Settings, monkeypatch
    ) -> None:  # noqa: ANN001
        """A browser is being navigated here, so JSON would leave the user stranded."""
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        async with await client_with(session_factory, oauth_settings) as http:
            state = _issued_state(
                (await http.get("/auth/github/start")).json()["authorize_url"]
            )
            response = await http.get(
                "/auth/github/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == "http://localhost:5173"
        assert oauth_settings.session_cookie_name in response.cookies

    async def test_token_is_stored_encrypted_and_never_returned(
        self, session_factory, oauth_settings: Settings, db: AsyncSession, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        async with await client_with(session_factory, oauth_settings) as http:
            state = _issued_state(
                (await http.get("/auth/github/start")).json()["authorize_url"]
            )
            response = await http.get(
                "/auth/github/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )

        assert "gho_exchanged" not in response.text

        connection = (
            await db.execute(
                select(GitHubConnection).join(User).where(User.github_login == "diksha")
            )
        ).scalar_one()

        assert connection.encrypted_token != "gho_exchanged"
        assert "gho_exchanged" not in connection.encrypted_token
        assert build_cipher(oauth_settings).decrypt(connection.encrypted_token) == "gho_exchanged"


class TestTokenSignIn:
    async def test_a_valid_token_signs_in_and_links_the_account(
        self, client: AsyncClient, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        response = await client.post(
            "/auth/dev-login", json={"github_token": "ghp_a_valid_looking_token"}
        )

        assert response.status_code == 200
        assert response.json()["github_login"] == "diksha"

        # The session now works on a protected route.
        assert (await client.get("/auth/me")).status_code == 200

    async def test_a_token_github_rejects_is_a_401(
        self, client: AsyncClient, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            auth_routes,
            "GitHubClient",
            _client_factory(github_transport(user_status=401)),
        )

        response = await client.post(
            "/auth/dev-login", json={"github_token": "ghp_not_a_real_token"}
        )

        assert response.status_code == 401

    async def test_too_short_a_token_is_rejected_before_calling_github(
        self, client: AsyncClient
    ) -> None:
        response = await client.post("/auth/dev-login", json={"github_token": "abc"})

        assert response.status_code == 422

    async def test_token_sign_in_is_unavailable_in_production(
        self, session_factory, settings: Settings
    ) -> None:  # noqa: ANN001
        production = settings.model_copy(
            update={"app_env": AppEnv.production, "session_secret": "a-real-secret"}
        )

        async with await client_with(session_factory, production) as http:
            response = await http.post(
                "/auth/dev-login", json={"github_token": "ghp_a_valid_looking_token"}
            )

        # 404 rather than 403: the route does not exist as far as production is concerned.
        assert response.status_code == 404


class TestSession:
    async def test_me_is_401_when_anonymous(self, client: AsyncClient) -> None:
        """The sign-in screen depends on this being 401 and not 500 or an empty 200."""
        assert (await client.get("/auth/me")).status_code == 401

    async def test_session_cookie_is_httponly_and_lax(
        self, client: AsyncClient, monkeypatch
    ) -> None:  # noqa: ANN001
        """HttpOnly is what stops page JavaScript reading the session after an XSS bug."""
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        response = await client.post(
            "/auth/dev-login", json={"github_token": "ghp_a_valid_looking_token"}
        )
        header = response.headers["set-cookie"].lower()

        assert "httponly" in header
        assert "samesite=lax" in header

    async def test_logout_clears_the_session(
        self, client: AsyncClient, monkeypatch
    ) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            auth_routes, "GitHubClient", _client_factory(github_transport())
        )

        await client.post("/auth/dev-login", json={"github_token": "ghp_a_valid_looking_token"})
        assert (await client.get("/auth/me")).status_code == 200

        assert (await client.post("/auth/logout")).status_code == 204
        assert (await client.get("/auth/me")).status_code == 401


# ------------------------------------------------------------------------------- helpers


def _client_factory(transport: httpx.MockTransport):  # noqa: ANN202
    """Patches GitHubClient so its httpx calls hit a mock transport instead of the network."""
    from app.github.client import GitHubClient

    class Patched(GitHubClient):
        def __init__(self, token: str, **kwargs) -> None:  # noqa: ANN003
            super().__init__(token, **kwargs)
            self._mock = transport

        async def _request(self, method: str, path: str, **kwargs):  # noqa: ANN003, ANN202
            async with httpx.AsyncClient(transport=self._mock, base_url="https://api.test") as c:
                response = await c.request(method, path, **kwargs)

                if response.status_code >= 400:
                    from app.github.client import GitHubError

                    raise GitHubError(response.status_code, "mock rejection")

                return response.json()

        async def exchange_oauth_code(self, **_kwargs):  # noqa: ANN003, ANN202
            return {"access_token": "gho_exchanged", "scope": "repo"}

    return Patched


def _issued_state(authorize_url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(authorize_url).query)["state"][0]
