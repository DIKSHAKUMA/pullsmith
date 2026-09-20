"""Creating a branch, committing and opening a pull request.

Commits are made through the GitHub **Contents API** rather than by pushing with the `git` CLI.
Three reasons:

* no credential ever has to be written into a git remote URL or a credential helper, so a token
  cannot leak into `.git/config` or a process listing;
* it works identically whether the workspace is local or in a remote sandbox;
* each file write is an explicit, auditable API call rather than an opaque push.

The cost is one request per changed file, which is acceptable for the small diffs this agent
produces and is a poor fit only for very large change sets.

Every function here assumes the caller has already passed the approval gate. Nothing in this
module checks it — that is `review_service.assert_push_allowed`'s job, and duplicating the check
would invite one copy drifting from the other.
"""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from app.github.client import GitHubClient, GitHubError

logger = logging.getLogger(__name__)


@dataclass
class CreatedPullRequest:
    number: int
    html_url: str
    branch: str
    head_sha: str


class PullRequestClient:
    def __init__(self, client: GitHubClient, *, owner: str, repository: str) -> None:
        self._client = client
        self._owner = owner
        self._repository = repository

    @property
    def _base(self) -> str:
        return f"/repos/{self._owner}/{self._repository}"

    async def default_branch_sha(self, branch: str) -> str:
        """The commit a new branch will start from."""
        data = await self._client._request(  # noqa: SLF001 - internal transport, one client
            "GET", f"{self._base}/git/ref/heads/{branch}"
        )
        return data["object"]["sha"]

    async def create_branch(self, *, branch: str, from_sha: str) -> None:
        try:
            await self._client._request(  # noqa: SLF001
                "POST",
                f"{self._base}/git/refs",
                json={"ref": f"refs/heads/{branch}", "sha": from_sha},
            )
        except GitHubError as exc:
            # 422 here means the ref already exists, which is safe to continue from: the
            # branch name includes the run id, so it belongs to this run.
            if exc.status_code != 422:
                raise

            logger.info("branch %s already exists, reusing it", branch)

    async def _existing_file_sha(self, path: str, branch: str) -> str | None:
        """The blob SHA of a file on a branch, or None if it does not exist.

        Required when updating: the Contents API needs the current SHA, which is also an
        optimistic-concurrency check — if someone else changed the file first, the update
        fails rather than silently overwriting their work.
        """
        try:
            data = await self._client._request(  # noqa: SLF001
                "GET", f"{self._base}/contents/{path}", params={"ref": branch}
            )
        except GitHubError as exc:
            if exc.status_code == 404:
                return None
            raise

        return data.get("sha") if isinstance(data, dict) else None

    async def commit_file(
        self, *, path: str, content: str, message: str, branch: str
    ) -> str:
        """Creates or updates one file. Returns the resulting commit SHA."""
        payload: dict[str, object] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }

        existing = await self._existing_file_sha(path, branch)

        if existing is not None:
            payload["sha"] = existing

        data = await self._client._request(  # noqa: SLF001
            "PUT", f"{self._base}/contents/{path}", json=payload
        )
        return data["commit"]["sha"]

    async def delete_file(self, *, path: str, message: str, branch: str) -> str | None:
        existing = await self._existing_file_sha(path, branch)

        if existing is None:
            return None

        data = await self._client._request(  # noqa: SLF001
            "DELETE",
            f"{self._base}/contents/{path}",
            json={"message": message, "sha": existing, "branch": branch},
        )
        return data["commit"]["sha"]

    async def open_pull_request(
        self, *, title: str, body: str, head: str, base: str
    ) -> CreatedPullRequest:
        data = await self._client._request(  # noqa: SLF001
            "POST",
            f"{self._base}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
        )

        return CreatedPullRequest(
            number=data["number"],
            html_url=data["html_url"],
            branch=head,
            head_sha=data["head"]["sha"],
        )


async def publish_changes(
    prs: PullRequestClient,
    *,
    workspace: Path,
    changed_paths: list[str],
    deleted_paths: list[str],
    branch: str,
    base_branch: str,
    commit_message: str,
    title: str,
    body: str,
) -> CreatedPullRequest:
    """Creates the branch, commits every change, and opens the pull request.

    Reads file contents from the workspace at push time rather than trusting anything cached, so
    what lands on the branch is exactly what was on disk when the diff was approved.
    """
    base_sha = await prs.default_branch_sha(base_branch)
    await prs.create_branch(branch=branch, from_sha=base_sha)

    last_sha: str | None = None

    for path in changed_paths:
        target = workspace / path

        if not target.is_file():
            logger.warning("skipping %s: no longer present in the workspace", path)
            continue

        last_sha = await prs.commit_file(
            path=path,
            content=target.read_text(encoding="utf-8", errors="replace"),
            message=f"{commit_message} ({path})",
            branch=branch,
        )

    for path in deleted_paths:
        sha = await prs.delete_file(
            path=path, message=f"{commit_message} (remove {path})", branch=branch
        )
        last_sha = sha or last_sha

    if last_sha is None:
        raise GitHubError(422, "nothing was committed; refusing to open an empty pull request")

    logger.info("committed %s file(s) to %s", len(changed_paths), branch)

    return await prs.open_pull_request(title=title, body=body, head=branch, base=base_branch)
