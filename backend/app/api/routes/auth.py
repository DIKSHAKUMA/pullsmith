"""Authentication: GitHub OAuth code flow, plus a development-only token login.

The session is an HttpOnly cookie rather than a token in localStorage so page
JavaScript cannot read it, which limits the blast radius of an XSS bug.
"""

import logging
import secrets

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import CurrentUser, DbDep, SettingsDep
from app.config.settings import AppEnv
from app.db.base import new_id
from app.github.client import GitHubClient, GitHubError
from app.models.core import GitHubConnection, User
from app.schemas.api import UserResponse
from app.security.crypto import build_cipher, build_session_codec

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# In-process CSRF state store for the OAuth round trip. Fine for a single-process
# deployment; a multi-instance deployment would move this to the database.
_oauth_states: set[str] = set()


class AuthStartResponse(BaseModel):
    authorize_url: str


class DevLoginRequest(BaseModel):
    github_token: str = Field(min_length=8, description="Personal access token, development only")


def _set_session_cookie(response: Response, settings: SettingsDep, user_id: str) -> None:
    token = build_session_codec(settings).issue(user_id)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.session_ttl_seconds,
        path="/",
    )


async def _upsert_user_and_connection(
    session: DbDep, settings: SettingsDep, *, profile: dict, token: str, scopes: str | None
) -> User:
    user = (
        await session.execute(select(User).where(User.github_user_id == profile["id"]))
    ).scalar_one_or_none()

    if user is None:
        user = User(
            id=new_id(),
            github_user_id=profile["id"],
            github_login=profile["login"],
            display_name=profile.get("name"),
            avatar_url=profile.get("avatar_url"),
        )
        session.add(user)
        await session.flush()
    else:
        user.github_login = profile["login"]
        user.display_name = profile.get("name")
        user.avatar_url = profile.get("avatar_url")

    encrypted = build_cipher(settings).encrypt(token)

    connection = (
        await session.execute(
            select(GitHubConnection).where(GitHubConnection.user_id == user.id)
        )
    ).scalar_one_or_none()

    if connection is None:
        session.add(
            GitHubConnection(
                id=new_id(), user_id=user.id, encrypted_token=encrypted, scopes=scopes
            )
        )
    else:
        connection.encrypted_token = encrypted
        connection.scopes = scopes

    await session.flush()
    return user


@router.get("/github/start", response_model=AuthStartResponse)
async def github_start(settings: SettingsDep) -> AuthStartResponse:
    if not settings.github_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured",
        )

    state = secrets.token_urlsafe(24)
    _oauth_states.add(state)

    query = (
        f"client_id={settings.github_client_id}"
        f"&redirect_uri={settings.github_oauth_redirect_uri}"
        f"&scope=repo%20read:user"
        f"&state={state}"
    )
    return AuthStartResponse(authorize_url=f"https://github.com/login/oauth/authorize?{query}")


@router.get("/github/callback")
async def github_callback(
    code: str,
    state: str,
    settings: SettingsDep,
    session: DbDep,
) -> RedirectResponse:
    """Completes the OAuth round trip and sends the browser back to the application.

    A redirect, not a JSON body. GitHub navigates the *browser* here, so returning
    `UserResponse` would leave the user staring at raw JSON on the API's origin with no way
    back. The session cookie is set on the redirect response, which the browser keeps.
    """
    # Discarding the state prevents replay and confirms the redirect started here.
    if state not in _oauth_states:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")
    _oauth_states.discard(state)

    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured",
        )

    exchange_client = GitHubClient("unused", api_base=settings.github_api_base)

    try:
        payload = await exchange_client.exchange_oauth_code(
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            code=code,
            redirect_uri=settings.github_oauth_redirect_uri,
        )
        token = payload["access_token"]
        profile = await GitHubClient(
            token, api_base=settings.github_api_base
        ).get_authenticated_user()
    except (GitHubError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="GitHub authentication failed"
        ) from exc

    user = await _upsert_user_and_connection(
        session, settings, profile=profile, token=token, scopes=payload.get("scope")
    )

    # 303 so the browser issues a GET for the destination regardless of how it arrived.
    redirect = RedirectResponse(
        url=settings.frontend_origin, status_code=status.HTTP_303_SEE_OTHER
    )
    _set_session_cookie(redirect, settings, user.id)

    logger.info("github oauth completed for %s", user.github_login)
    return redirect


@router.post("/dev-login", response_model=UserResponse)
async def dev_login(
    body: DevLoginRequest,
    response: Response,
    settings: SettingsDep,
    session: DbDep,
) -> UserResponse:
    """Signs in with a personal access token. Blocked outside development."""
    if settings.app_env is AppEnv.production:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    try:
        profile = await GitHubClient(
            body.github_token, api_base=settings.github_api_base
        ).get_authenticated_user()
    except GitHubError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token rejected by GitHub"
        ) from exc

    user = await _upsert_user_and_connection(
        session, settings, profile=profile, token=body.github_token, scopes="pat"
    )

    _set_session_cookie(response, settings, user.id)
    return UserResponse.model_validate(user)


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response, settings: SettingsDep) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")
