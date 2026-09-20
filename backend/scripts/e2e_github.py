"""Sets up an end-to-end run against a **real GitHub repository**, so a real pull request opens.

This is the one thing `e2e_setup.py` cannot prove. That script uses a local fixture repository,
so the publish path never runs. Here the repository is real, the issue is real, the clone is over
HTTPS from github.com, and approving the diff opens an actual pull request.

No OAuth app required. A fine-grained personal access token is enough, and the token is stored
exactly the way an OAuth token would be: Fernet-encrypted in `github_connection`, decrypted only
inside the worker, never logged.

Prerequisites:

1. A **throwaway public** repository you own. Public matters: the agent clones with plain `git`
   and no credentials, deliberately, so a token can never leak into `.git/config` or a process
   listing. Private repositories are therefore not supported yet.
2. A token with **Contents: read and write** and **Issues: read and write** on that repository,
   plus **Pull requests: read and write**.
   Create one at https://github.com/settings/personal-access-tokens/new
3. The token in the environment, not on the command line, so it stays out of shell history:

   ```powershell
   $env:GITHUB_PAT = "github_pat_..."
   python -m scripts.e2e_github --repo yourname/agent-sandbox --seed
   ```

`--seed` writes the buggy fixture files and opens the issue in the repository. Run it once, then
drop the flag on later runs.
"""

import argparse
import asyncio
import getpass
import os
import sys

from sqlalchemy import select

from app.config.settings import get_settings
from app.db.base import Base, new_id
from app.db.session import init_engine, session_scope
from app.github.client import GitHubClient, GitHubError
from app.github.pr import PullRequestClient
from app.models.core import GitHubConnection, Issue, Repository, User
from app.security.crypto import build_cipher, build_session_codec
from app.services import run_manager
from scripts import scenarios
from scripts.e2e_setup import check_configuration, reset_previous_runs


def read_token() -> str:
    """Reads the token from the environment, or prompts without echoing it."""
    token = os.environ.get("GITHUB_PAT", "").strip()

    if token:
        return token

    # getpass keeps it out of the terminal scrollback as well as shell history.
    return getpass.getpass("GitHub personal access token (not echoed): ").strip()


async def seed_repository(
    client: GitHubClient,
    *,
    owner: str,
    name: str,
    branch: str,
    scenario: scenarios.Scenario,
) -> None:
    """Writes the scenario's buggy files onto the default branch via the Contents API.

    The same API the agent uses to publish, so this also proves the token has the write
    permission the pull request will need, before an agent run is spent finding out.
    """
    prs = PullRequestClient(client, owner=owner, repository=name)

    for path, content in scenario.files.items():
        await prs.commit_file(
            path=path,
            content=content,
            message=f"Add {path} for agent end-to-end testing",
            branch=branch,
        )
        print(f"  committed {path}")


async def ensure_issue(
    client: GitHubClient, *, owner: str, name: str, scenario: scenarios.Scenario
) -> dict:
    """Finds the scenario's issue or opens it. Never opens a duplicate."""
    existing = await client.list_issues(owner, name, state="open", per_page=100)

    for issue in existing:
        if issue["title"] == scenario.issue_title:
            print(f"  reusing issue #{issue['number']}")
            return issue

    created = await client._request(  # noqa: SLF001 - one client, internal transport
        "POST",
        f"/repos/{owner}/{name}/issues",
        json={
            "title": scenario.issue_title,
            "body": scenario.issue_body,
            "labels": ["bug"],
        },
    )
    print(f"  opened issue #{created['number']}")
    return created


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name of a throwaway public repo")
    parser.add_argument(
        "--seed", action="store_true", help="write the fixture files and open the issue"
    )
    parser.add_argument(
        "--reset", action="store_true", help="discard previous runs for the fixture issue"
    )
    parser.add_argument(
        "--require-plan-approval",
        action="store_true",
        help="stop for human review of the plan before any code is written",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="prepare everything but do not queue a run, so you can start one from the UI",
    )
    parser.add_argument(
        "--scenario",
        default=scenarios.DEFAULT_SCENARIO,
        choices=sorted(scenarios.SCENARIOS),
        help="which bug to plant. Run both to see the agent is reading the issue.",
    )
    arguments = parser.parse_args()

    scenario = scenarios.get(arguments.scenario)

    if "/" not in arguments.repo:
        raise SystemExit("--repo must be owner/name, for example diksha/agent-sandbox")

    owner, name = arguments.repo.split("/", 1)

    settings = get_settings()
    problems = check_configuration(settings)

    if problems:
        print("Cannot start a live run yet:\n")
        for problem in problems:
            print(f"  - {problem}\n")
        raise SystemExit(1)

    token = read_token()

    if not token:
        raise SystemExit("No token supplied. Set GITHUB_PAT or enter it at the prompt.")

    client = GitHubClient(token, api_base=settings.github_api_base)

    try:
        profile = await client.get_authenticated_user()
    except GitHubError as exc:
        raise SystemExit(f"GitHub rejected the token: {exc}") from exc

    print(f"authenticated as {profile['login']}")

    try:
        repository_payload = await client.get_repository(owner, name)
    except GitHubError as exc:
        raise SystemExit(
            f"Cannot read {arguments.repo}: {exc}\n"
            f"Check the repository exists and the token grants it Contents access."
        ) from exc

    branch = repository_payload["default_branch"]

    if repository_payload.get("private"):
        raise SystemExit(
            f"{arguments.repo} is private. The agent clones with plain git and no credentials "
            f"on purpose, so a token cannot leak into .git/config or a process listing. Use a "
            f"public throwaway repository, or implement authenticated cloning first."
        )

    print(f"repository {repository_payload['full_name']} on branch {branch}")

    print(f"scenario: {scenario.key} - {scenario.summary}")

    if arguments.seed:
        print("seeding fixture files:")
        await seed_repository(
            client, owner=owner, name=name, branch=branch, scenario=scenario
        )

    issue_payload = await ensure_issue(client, owner=owner, name=name, scenario=scenario)

    engine = init_engine(settings)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_scope() as session:
        user = (
            await session.execute(
                select(User).where(User.github_user_id == profile["id"])
            )
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

        encrypted = build_cipher(settings).encrypt(token)

        connection_row = (
            await session.execute(
                select(GitHubConnection).where(GitHubConnection.user_id == user.id)
            )
        ).scalar_one_or_none()

        if connection_row is None:
            session.add(
                GitHubConnection(
                    id=new_id(), user_id=user.id, encrypted_token=encrypted, scopes="pat"
                )
            )
        else:
            connection_row.encrypted_token = encrypted
            connection_row.scopes = "pat"

        repository = (
            await session.execute(
                select(Repository).where(
                    Repository.user_id == user.id,
                    Repository.github_repo_id == repository_payload["id"],
                )
            )
        ).scalar_one_or_none()

        if repository is None:
            repository = Repository(
                id=new_id(),
                user_id=user.id,
                github_repo_id=repository_payload["id"],
                owner=owner,
                name=name,
                full_name=repository_payload["full_name"],
            )
            session.add(repository)

        repository.default_branch = branch
        repository.clone_url = repository_payload["clone_url"]
        repository.primary_language = repository_payload.get("language")
        repository.is_private = bool(repository_payload.get("private"))
        await session.flush()

        issue = (
            await session.execute(
                select(Issue).where(
                    Issue.repository_id == repository.id,
                    Issue.number == issue_payload["number"],
                )
            )
        ).scalar_one_or_none()

        if issue is None:
            issue = Issue(
                id=new_id(),
                repository_id=repository.id,
                number=issue_payload["number"],
                title=issue_payload["title"],
                body=issue_payload.get("body"),
                state=issue_payload.get("state", "open"),
                labels=["bug"],
                html_url=issue_payload.get("html_url"),
            )
            session.add(issue)
            await session.flush()

        if arguments.reset:
            removed = await reset_previous_runs(session, issue.id)
            print(f"discarded {removed} previous run(s)")

        run = None

        if not arguments.no_run:
            run = await run_manager.create_run(
                session,
                user_id=user.id,
                repository_id=repository.id,
                issue_id=issue.id,
                max_iterations=settings.max_iterations,
                require_plan_approval=arguments.require_plan_approval,
            )

        await session.commit()

        cookie = build_session_codec(settings).issue(user.id)

    print()
    print(f"ISSUE={issue_payload['html_url']}")
    print(f"COOKIE={settings.session_cookie_name}={cookie}")
    print(f"should fail before the fix: {', '.join(scenario.expected_failures)}")

    if run is None:
        print()
        print("No run queued (--no-run). Start one yourself:")
        print("  1. start the worker:  python -m app.worker.main")
        print("  2. start the API:     python -m uvicorn app.main:app --reload")
        print("  3. start the UI:      cd ../frontend; npm run dev")
        print("  4. open http://localhost:5173, sign in with the same token,")
        print(f"     link {repository_payload['full_name']}, sync issues, and start a run.")
        return

    print(f"RUN_ID={run.id}")
    print()
    print("Start the worker, then approve to open a real pull request:")
    print(r"    .\venv\Scripts\Activate.ps1")
    print("    python -m app.worker.main")
    print(f"    python -m scripts.e2e_watch {run.id} --approve")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
