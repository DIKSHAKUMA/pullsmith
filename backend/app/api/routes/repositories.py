"""Repository and issue endpoints.

GitHub is the source of truth. Repositories and issues are mirrored locally only so
runs, snapshots and events can reference stable internal ids.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbDep, GitHubDep
from app.db.base import new_id
from app.github.client import GitHubError
from app.models.core import Issue, Repository
from app.schemas.api import (
    GitHubRepositoryOption,
    IssueResponse,
    LinkRepositoryRequest,
    RepositoryResponse,
    SyncIssuesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["repositories"])


@router.get("/github/repositories", response_model=list[GitHubRepositoryOption])
async def list_github_repositories(github: GitHubDep) -> list[GitHubRepositoryOption]:
    try:
        raw = await github.list_repositories()
    except GitHubError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return [
        GitHubRepositoryOption(
            github_repo_id=item["id"],
            owner=item["owner"]["login"],
            name=item["name"],
            full_name=item["full_name"],
            default_branch=item.get("default_branch") or "main",
            primary_language=item.get("language"),
            is_private=bool(item.get("private")),
        )
        for item in raw
    ]


@router.post(
    "/repositories", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED
)
async def link_repository(
    body: LinkRepositoryRequest, user: CurrentUser, session: DbDep, github: GitHubDep
) -> RepositoryResponse:
    try:
        data = await github.get_repository(body.owner, body.name)
    except GitHubError as exc:
        code = (
            status.HTTP_404_NOT_FOUND
            if exc.status_code == 404
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    existing = (
        await session.execute(
            select(Repository).where(
                Repository.user_id == user.id, Repository.github_repo_id == data["id"]
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        return RepositoryResponse.model_validate(existing)

    repository = Repository(
        id=new_id(),
        user_id=user.id,
        github_repo_id=data["id"],
        owner=data["owner"]["login"],
        name=data["name"],
        full_name=data["full_name"],
        default_branch=data.get("default_branch") or "main",
        primary_language=data.get("language"),
        is_private=bool(data.get("private")),
        clone_url=data.get("clone_url"),
    )
    session.add(repository)
    await session.flush()

    return RepositoryResponse.model_validate(repository)


@router.get("/repositories", response_model=list[RepositoryResponse])
async def list_repositories(user: CurrentUser, session: DbDep) -> list[RepositoryResponse]:
    result = await session.execute(
        select(Repository)
        .where(Repository.user_id == user.id)
        .order_by(Repository.created_at.desc())
    )
    return [RepositoryResponse.model_validate(row) for row in result.scalars()]


async def _owned_repository(session: DbDep, user: CurrentUser, repository_id: str) -> Repository:
    repository = (
        await session.execute(
            select(Repository).where(
                Repository.id == repository_id, Repository.user_id == user.id
            )
        )
    ).scalar_one_or_none()

    if repository is None:
        # 404 rather than 403 so the endpoint does not confirm that another user's
        # repository exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    return repository


@router.get("/repositories/{repository_id}", response_model=RepositoryResponse)
async def get_repository(
    repository_id: str, user: CurrentUser, session: DbDep
) -> RepositoryResponse:
    repository = await _owned_repository(session, user, repository_id)
    return RepositoryResponse.model_validate(repository)


@router.post("/repositories/{repository_id}/issues/sync", response_model=SyncIssuesResponse)
async def sync_issues(
    repository_id: str, user: CurrentUser, session: DbDep, github: GitHubDep
) -> SyncIssuesResponse:
    repository = await _owned_repository(session, user, repository_id)

    try:
        raw = await github.list_issues(repository.owner, repository.name)
    except GitHubError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    stored: list[Issue] = []

    for item in raw:
        issue = (
            await session.execute(
                select(Issue).where(
                    Issue.repository_id == repository.id, Issue.number == item["number"]
                )
            )
        ).scalar_one_or_none()

        labels = [label["name"] for label in item.get("labels", []) if isinstance(label, dict)]

        if issue is None:
            issue = Issue(
                id=new_id(),
                repository_id=repository.id,
                number=item["number"],
                title=item["title"],
                body=item.get("body"),
                state=item.get("state", "open"),
                labels=labels,
                html_url=item.get("html_url"),
            )
            session.add(issue)
        else:
            issue.title = item["title"]
            issue.body = item.get("body")
            issue.state = item.get("state", "open")
            issue.labels = labels

        stored.append(issue)

    await session.flush()

    return SyncIssuesResponse(
        synced=len(stored),
        issues=[IssueResponse.model_validate(issue) for issue in stored],
    )


@router.get("/repositories/{repository_id}/issues", response_model=list[IssueResponse])
async def list_issues(
    repository_id: str, user: CurrentUser, session: DbDep
) -> list[IssueResponse]:
    await _owned_repository(session, user, repository_id)

    result = await session.execute(
        select(Issue).where(Issue.repository_id == repository_id).order_by(Issue.number.desc())
    )
    return [IssueResponse.model_validate(row) for row in result.scalars()]
