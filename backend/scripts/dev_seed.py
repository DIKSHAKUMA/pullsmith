"""Seeds a development user, repository and issue, and prints a session cookie.

Development helper only. It creates local fixture data so the API and worker can be
exercised end to end before GitHub OAuth is configured. It never contacts GitHub and
the token it stores is a placeholder.

Usage:  python -m scripts.dev_seed
"""

import asyncio

from sqlalchemy import select

from app.config.settings import AppEnv, get_settings
from app.db.base import Base, new_id
from app.db.session import init_engine, session_scope
from app.models.core import GitHubConnection, Issue, Repository, User
from app.security.crypto import build_cipher, build_session_codec


async def main() -> None:
    settings = get_settings()

    if settings.app_env is AppEnv.production:
        raise SystemExit("dev_seed refuses to run in production")

    engine = init_engine(settings)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_scope() as session:
        user = (
            await session.execute(select(User).where(User.github_login == "dev-user"))
        ).scalar_one_or_none()

        if user is None:
            user = User(
                id=new_id(),
                github_user_id=1,
                github_login="dev-user",
                display_name="Development User",
            )
            session.add(user)
            await session.flush()

            session.add(
                GitHubConnection(
                    id=new_id(),
                    user_id=user.id,
                    encrypted_token=build_cipher(settings).encrypt("placeholder-not-a-real-token"),
                    scopes="dev",
                )
            )

        repository = (
            await session.execute(
                select(Repository).where(
                    Repository.user_id == user.id, Repository.full_name == "dev-user/demo-repo"
                )
            )
        ).scalar_one_or_none()

        if repository is None:
            repository = Repository(
                id=new_id(),
                user_id=user.id,
                github_repo_id=1001,
                owner="dev-user",
                name="demo-repo",
                full_name="dev-user/demo-repo",
                default_branch="main",
                primary_language="Python",
            )
            session.add(repository)
            await session.flush()

        issue = (
            await session.execute(
                select(Issue).where(Issue.repository_id == repository.id, Issue.number == 1)
            )
        ).scalar_one_or_none()

        if issue is None:
            issue = Issue(
                id=new_id(),
                repository_id=repository.id,
                number=1,
                title="Profile update returns 500 when email is empty",
                body="Expected a validation error, received HTTP 500.",
                state="open",
                labels=["bug"],
            )
            session.add(issue)
            await session.flush()

        await session.commit()

        cookie = build_session_codec(settings).issue(user.id)

        print(f"USER_ID={user.id}")
        print(f"REPOSITORY_ID={repository.id}")
        print(f"ISSUE_ID={issue.id}")
        print(f"COOKIE={settings.session_cookie_name}={cookie}")


if __name__ == "__main__":
    asyncio.run(main())
