"""GitHub REST client.

Every call has an explicit timeout and a narrow retry policy. Retrying is limited to
transient conditions (timeouts, 5xx, secondary rate limits); a 401 or 403 is a real
answer and retrying it only wastes the rate-limit budget.
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GitHubError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"GitHub API error {status_code}: {message}")
        self.status_code = status_code


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        api_base: str = "https://api.github.com",
        timeout: float = 15.0,
        max_attempts: int = 3,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pullsmith",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._api_base}{path}"
        delay = 1.0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    response = await client.request(
                        method, url, headers=self._headers(), params=params, json=json
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == self._max_attempts:
                        raise GitHubError(504, f"transport failure: {type(exc).__name__}") from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if response.status_code in RETRYABLE_STATUS and attempt < self._max_attempts:
                    # Honour Retry-After when GitHub tells us how long to wait.
                    wait = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "github %s %s returned %s, retrying in %.1fs",
                        method,
                        path,
                        response.status_code,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue

                if response.status_code >= 400:
                    detail = response.json().get("message", "unknown") if response.content else ""
                    raise GitHubError(response.status_code, str(detail))

                return response.json() if response.content else None

        raise GitHubError(500, "request loop exhausted")

    async def get_authenticated_user(self) -> dict[str, Any]:
        return await self._request("GET", "/user")

    async def list_repositories(
        self, *, max_pages: int = 5, per_page: int = 100
    ) -> list[dict[str, Any]]:
        """Every repository the user can push to, following pagination.

        Two details that were wrong before and are worth stating:

        **It pages.** A single request returns at most 100 repositories, so a user with more than
        that silently lost the rest. A list that quietly omits the repository you were looking
        for is worse than an error, because there is nothing to tell you it happened.

        **``affiliation`` includes collaborator.** Owner-only hid every repository the user has
        write access to but does not own, which is most shared work. Safety does not come from
        hiding them: a repository still has to be linked deliberately, and no pull request opens
        without a human approving the diff.

        ``max_pages`` bounds it so a user with thousands of repositories cannot make one page
        load hang on 40 sequential API calls.
        """
        collected: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            batch = await self._request(
                "GET",
                "/user/repos",
                params={
                    "page": page,
                    "per_page": per_page,
                    "sort": "updated",
                    "affiliation": "owner,collaborator",
                },
            )

            if not batch:
                break

            collected.extend(batch)

            # A short page is the last page, so stop rather than spend a request proving it.
            if len(batch) < per_page:
                break

        logger.info("listed %s repositories across %s page(s)", len(collected), page)
        return collected

    async def get_repository(self, owner: str, name: str) -> dict[str, Any]:
        return await self._request("GET", f"/repos/{owner}/{name}")

    async def list_issues(
        self, owner: str, name: str, *, state: str = "open", per_page: int = 30
    ) -> list[dict[str, Any]]:
        issues = await self._request(
            "GET",
            f"/repos/{owner}/{name}/issues",
            params={"state": state, "per_page": per_page},
        )
        # The issues endpoint also returns pull requests; they are not work items here.
        return [issue for issue in issues if "pull_request" not in issue]

    async def get_issue(self, owner: str, name: str, number: int) -> dict[str, Any]:
        return await self._request("GET", f"/repos/{owner}/{name}/issues/{number}")

    async def exchange_oauth_code(
        self, *, client_id: str, client_secret: str, code: str, redirect_uri: str
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )

        if response.status_code >= 400:
            raise GitHubError(response.status_code, "oauth code exchange failed")

        payload: dict[str, Any] = response.json()

        if "error" in payload:
            raise GitHubError(400, str(payload.get("error_description", payload["error"])))

        return payload
