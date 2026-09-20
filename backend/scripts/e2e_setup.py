"""Sets up a real end-to-end run against a local git repository.

This is the honest smoke test. Everything is real except the repository, which is created here
so the run costs nothing and repeats identically:

* a **real git repository** with a real bug and a test that fails because of it
* the **real Gemini model** doing analysis, planning and implementation
* the **real tool layer**, safety boundaries, sandbox runner and test runner
* the **real** diff, risk score, approval gate and Postgres state machine

What it does not do: open a pull request. That needs a GitHub OAuth app and a real token, and
the run will park at WAITING_FOR_APPROVAL instead.

Usage:
    python -m scripts.e2e_setup            # create the fixture and queue a run
    python -m scripts.e2e_setup --reset    # discard previous runs for the issue first

Then start the worker in another terminal and watch it work.
"""

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy import delete, select

from app.config.settings import AppEnv, SandboxBackend, get_settings
from app.db.base import Base, new_id
from app.db.session import init_engine, session_scope
from app.models.core import AgentEvent, AgentRun, GitHubConnection, Issue, Job, Repository, User
from app.security.crypto import build_cipher, build_session_codec
from app.services import run_manager

FIXTURE_NAME = "e2e-fixture-repo"

# The bug: no guard on the divisor, so the caller gets ZeroDivisionError instead of a clear
# domain error. Small enough for a model to fix, real enough that a test can prove it.
BROKEN_SOURCE = '''\
"""Arithmetic helpers used by the billing calculator."""


def divide(numerator: float, denominator: float) -> float:
    """Divides two numbers.

    Raises:
        ValueError: if the denominator is zero.
    """
    return numerator / denominator


def average(values: list[float]) -> float:
    """Returns the mean of a list of numbers."""
    return divide(sum(values), len(values))
'''

# test_average_of_empty_list_raises_value_error fails on the broken code, which is what gives
# the agent an objective signal rather than an opinion.
# An empty root conftest.py is what makes `pytest -q` work from the repository root. The
# console script does not put the working directory on sys.path (unlike `python -m pytest`),
# so without this `from calculator import ...` fails to import and every test errors during
# collection. Real repositories solve this with a conftest, a src layout plus an installed
# package, or pyproject's pythonpath; the fixture has to do the same to be realistic.
CONFTEST_SOURCE = '"""Makes the repository root importable for pytest."""\n'

TEST_SOURCE = '''\
import pytest

from calculator import average, divide


def test_divide_works():
    assert divide(10, 2) == 5


def test_average_works():
    assert average([2, 4, 6]) == 4


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_average_of_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        average([])
'''

ISSUE_TITLE = "average() and divide() raise ZeroDivisionError instead of ValueError"

ISSUE_BODY = """\
Calling `average([])` crashes with `ZeroDivisionError: division by zero`.

The docstring on `divide` says it raises `ValueError` when the denominator is zero, but there is
no check, so the raw `ZeroDivisionError` reaches the caller. Our API layer maps `ValueError` to
a 422 and anything else to a 500, so an empty list currently returns a 500.

Expected: `divide(1, 0)` and `average([])` both raise `ValueError` with a clear message.
Actual: both raise `ZeroDivisionError`.

The tests in `tests/test_calculator.py` cover this and are currently failing.
"""


def _git(*args: str, cwd: Path) -> None:
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=False,
        capture_output=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:300]}"
        )


def build_fixture_repository(root: Path) -> Path:
    """Creates a git repository with the bug committed, so the agent has real history."""
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    (root / "tests").mkdir(parents=True)

    (root / "calculator.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    (root / "conftest.py").write_text(CONFTEST_SOURCE, encoding="utf-8")
    (root / "tests" / "test_calculator.py").write_text(TEST_SOURCE, encoding="utf-8")

    # repo_map reads manifests, not an LLM, to decide how to run tests. pytest here is what
    # makes it detect `pytest -q`.
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Billing calculator\n\nArithmetic helpers.\n\nRun tests with `pytest -q`.\n",
        encoding="utf-8",
    )

    _git("init", "-b", "main", cwd=root)
    _git("config", "user.email", "fixture@example.test", cwd=root)
    _git("config", "user.name", "E2E Fixture", cwd=root)
    _git("add", ".", cwd=root)
    _git("commit", "-m", "Add billing calculator helpers", cwd=root)

    return root


def check_configuration(settings) -> list[str]:  # noqa: ANN001
    """Reports what would stop a live run, rather than letting it fail halfway through."""
    problems: list[str] = []

    if settings.app_env is AppEnv.production:
        problems.append("APP_ENV=production: this script refuses to touch a production database")

    if settings.sandbox_backend is not SandboxBackend.local_unsafe:
        problems.append(
            f"SANDBOX_BACKEND={settings.sandbox_backend} is not implemented, so the worker will "
            f"refuse to start. Set SANDBOX_BACKEND=local_unsafe in backend/.env, understanding "
            f"that it is process confinement and not isolation."
        )

    if not settings.gemini_api_key:
        problems.append("GEMINI_API_KEY is required: the agent cannot plan without a model")

    if settings.embedding_provider != "fake":
        problems.append(
            f"EMBEDDING_PROVIDER={settings.embedding_provider}. If the Gemini embedding quota is "
            f"exhausted, indexing will fail. Set EMBEDDING_PROVIDER=fake for a smoke test that "
            f"does not depend on quota (retrieval quality will be meaningless, which is why no "
            f"Recall@K is claimed)."
        )

    if shutil.which("pytest") is None:
        problems.append(
            "pytest is not on PATH. The local sandbox passes PATH through to the child, so the "
            "agent cannot run the fixture's tests. Activate the venv before starting the worker: "
            r".\venv\Scripts\Activate.ps1"
        )

    return problems


async def reset_previous_runs(session, issue_id: str) -> int:  # noqa: ANN001
    """Removes earlier runs for this issue so the script can be run repeatedly.

    `create_run` refuses a second active run per issue on purpose, which is correct behaviour
    and inconvenient for a demo, so the demo cleans up after itself instead.
    """
    run_ids = list(
        (await session.execute(select(AgentRun.id).where(AgentRun.issue_id == issue_id)))
        .scalars()
        .all()
    )

    if not run_ids:
        return 0

    await session.execute(delete(Job).where(Job.run_id.in_(run_ids)))
    await session.execute(delete(AgentEvent).where(AgentEvent.run_id.in_(run_ids)))
    await session.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))

    return len(run_ids)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="discard previous runs for the fixture issue"
    )
    parser.add_argument(
        "--require-plan-approval",
        action="store_true",
        help="stop for human review of the plan before any code is written",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    problems = check_configuration(settings)

    if problems:
        print("Cannot start a live run yet:\n")
        for problem in problems:
            print(f"  - {problem}\n")
        raise SystemExit(1)

    fixture = build_fixture_repository(Path(FIXTURE_NAME).resolve())
    print(f"fixture repository: {fixture}")

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
                    # A placeholder. Pull-request creation will fail with a GitHub 401, which is
                    # the honest outcome until a real OAuth app is configured.
                    encrypted_token=build_cipher(settings).encrypt("placeholder-not-a-real-token"),
                    scopes="dev",
                )
            )

        repository = (
            await session.execute(
                select(Repository).where(
                    Repository.user_id == user.id,
                    Repository.full_name == f"dev-user/{FIXTURE_NAME}",
                )
            )
        ).scalar_one_or_none()

        if repository is None:
            repository = Repository(
                id=new_id(),
                user_id=user.id,
                github_repo_id=2001,
                owner="dev-user",
                name=FIXTURE_NAME,
                full_name=f"dev-user/{FIXTURE_NAME}",
                default_branch="main",
                primary_language="Python",
            )
            session.add(repository)

        # Always repoint at the freshly built fixture: the commit SHA changed.
        repository.clone_url = str(fixture)
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
                title=ISSUE_TITLE,
                body=ISSUE_BODY,
                state="open",
                labels=["bug"],
            )
            session.add(issue)
            await session.flush()
        else:
            issue.title = ISSUE_TITLE
            issue.body = ISSUE_BODY

        if arguments.reset:
            removed = await reset_previous_runs(session, issue.id)
            print(f"discarded {removed} previous run(s)")

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
    print(f"RUN_ID={run.id}")
    print(f"COOKIE={settings.session_cookie_name}={cookie}")
    print()
    print("A job is queued. Start the worker to execute it:")
    print(r"    .\venv\Scripts\Activate.ps1")
    print("    python -m app.worker.main")
    print()
    print("Watch progress with:")
    print(f"    python -m scripts.e2e_watch {run.id}")
    print()
    print("Or open the dashboard at http://localhost:5173 with the API and frontend running.")


if __name__ == "__main__":
    # No WindowsSelectorEventLoopPolicy: it silences an asyncpg shutdown warning and cannot
    # spawn subprocesses, which any script that shells out to git needs.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
