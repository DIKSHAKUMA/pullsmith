"""Drives one run to a **real pull request**, with the model scripted.

### What this proves, and what it does not

The publish path is the last untested link in the chain, and it does not involve the model at
all. Waiting for a model quota to reset in order to test a GitHub API call would be silly, so
here the model is the same scripted provider the test suite uses and **everything else is real**:

| | |
|---|---|
| clone from github.com over HTTPS | real |
| file scan, repository map, tree-sitter chunking | real |
| pgvector write on Neon | real (fake embedding vectors) |
| tool layer, safety checks, `replace_in_file` | real |
| `pytest -q` in the sandbox runner | real |
| diff from recorded originals, risk score | real |
| approval gate and its hash binding | real |
| branch, commits, pull request on GitHub | **real** |
| the model choosing what to do | **scripted** |

So this proves the *plumbing* end to end. It does **not** prove the agent can solve a bug: that
needs a live model, and is what `e2e_github.py` is for. Neither script replaces the other.

Usage:
    $env:GITHUB_PAT = "github_pat_..."
    python -m scripts.e2e_publish --repo DIKSHAKUMA/agent-sandbox
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.agent.orchestrator import Orchestrator, create_pull_request
from app.agent.runtime import AgentRuntime, build_embeddings, build_registry, build_sandbox
from app.agent.states import RunState
from app.config.settings import get_settings
from app.db.session import init_engine, session_scope
from app.github import tokens
from app.llm.fake import FakeLLMProvider, ScriptedTurn, text_turn, tool_turn
from app.models.core import AgentRun, Repository
from app.models.review import PullRequest
from app.services import review_service

# The exact text to replace. Taken from the committed fixture, because `replace_in_file` refuses
# a snippet that does not appear exactly once — which is the guard that makes the tool safe and
# also means this string cannot drift from the repository.
FIND_TEXT = '''\
    Raises:
        ValueError: if the denominator is zero.
    """
    return numerator / denominator'''

REPLACE_TEXT = '''\
    Raises:
        ValueError: if the denominator is zero.
    """
    if denominator == 0:
        raise ValueError("denominator must not be zero")

    return numerator / denominator'''


def scripted_turns() -> list[ScriptedTurn]:
    """Exactly the turns one clean run consumes, in order.

    The fake provider raises if the code asks for one more than is written, so a change in the
    orchestrator's call pattern shows up here immediately rather than silently.
    """
    analysis = {
        "problem_summary": "divide() raises ZeroDivisionError instead of the documented ValueError",
        "expected_behaviour": "divide(1, 0) and average([]) raise ValueError",
        "actual_behaviour": "both raise ZeroDivisionError, which the API maps to a 500",
        "acceptance_criteria": ["tests/test_calculator.py passes"],
        "error_messages": ["ZeroDivisionError: division by zero"],
        "referenced_symbols": ["divide", "average"],
        "referenced_paths": ["calculator.py"],
        "search_queries": ["divide by zero guard", "average of empty list"],
        "is_actionable": True,
    }

    plan = {
        "problem_understanding": (
            "divide() documents a ValueError for a zero denominator but never checks for one, so "
            "the raw ZeroDivisionError reaches the caller and becomes a 500."
        ),
        "relevant_files": ["calculator.py", "tests/test_calculator.py"],
        "suspected_root_cause": "No validation of the denominator before the division in divide().",
        "root_cause_confidence": "high",
        "proposed_changes": [
            {
                "path": "calculator.py",
                "intent": "Raise ValueError when the denominator is zero, as the docstring states",
                "is_new_file": False,
            }
        ],
        "tests_to_add_or_update": ["tests/test_calculator.py"],
        "risks": ["Callers that currently catch ZeroDivisionError would need updating"],
        "verification_strategy": "Run pytest on tests/test_calculator.py, then the full suite.",
        "out_of_scope": ["Validating the types of the inputs"],
    }

    verification = {
        "issue_addressed": True,
        "tests_pass": True,
        "unrelated_changes_detected": False,
        "concerns": [],
        "summary": (
            "Added a zero check to divide(). average([]) now raises ValueError through divide()."
        ),
        "confidence": "high",
    }

    return [
        text_turn(json.dumps(analysis)),                      # analyse the issue
        text_turn("I have read calculator.py and the tests."),  # plan: tool loop ends
        text_turn(json.dumps(plan)),                           # plan: structured output
        tool_turn(                                             # implement: one targeted edit
            "replace_in_file",
            {
                "path": "calculator.py",
                "find": FIND_TEXT,
                "replace": REPLACE_TEXT,
                "reason": "Guard against a zero denominator",
            },
        ),
        text_turn("Added the guard to divide()."),             # implement: loop ends
        text_turn("The diff matches the plan."),               # verify: tool loop ends
        text_turn(json.dumps(verification)),                   # verify: structured output
    ]


def build_scripted_runtime(settings) -> AgentRuntime:  # noqa: ANN001
    """Production wiring with one substitution: the model."""
    return AgentRuntime(
        settings=settings,
        llm=FakeLLMProvider(scripted_turns(), model="scripted-for-publish-test"),
        embeddings=build_embeddings(settings),
        sandbox=build_sandbox(settings),
        registry=build_registry(),
        workspace_root=Path(settings.agent_workspace_root),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name, must match the seeded run")
    arguments = parser.parse_args()

    settings = get_settings()
    init_engine(settings)

    # Scoping to the named repository means this cannot accidentally publish a run belonging to
    # some other repository that happens to be more recent.
    repository_full_name = arguments.repo

    runtime = build_scripted_runtime(settings)

    print(f"model:   {runtime.llm.model} (scripted)")
    print(f"sandbox: {runtime.sandbox.name} (isolation: {runtime.sandbox.isolation})")
    print()

    async with session_scope() as session:
        # The most recent run for this repository, whatever state it stalled in.
        run = (
            await session.execute(
                select(AgentRun)
                .join(Repository, Repository.id == AgentRun.repository_id)
                .where(Repository.full_name == repository_full_name)
                .order_by(AgentRun.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if run is None:
            raise SystemExit(
                f"No run found for {repository_full_name}. Run scripts.e2e_github first."
            )

        # A run that failed on quota is restarted from the beginning: its workspace is gone and
        # the state machine has no path out of FAILED, by design.
        if RunState(run.state) is RunState.FAILED:
            print(f"previous run {run.id[:8]} is FAILED ({run.failure_category}); starting fresh")

            from sqlalchemy import delete

            from app.models.core import AgentEvent, Issue, Job
            from app.services import run_manager

            issue = (
                await session.execute(select(Issue).where(Issue.id == run.issue_id))
            ).scalar_one()

            await session.execute(delete(Job).where(Job.run_id == run.id))
            await session.execute(delete(AgentEvent).where(AgentEvent.run_id == run.id))
            await session.execute(delete(AgentRun).where(AgentRun.id == run.id))

            run = await run_manager.create_run(
                session,
                user_id=run.user_id,
                repository_id=run.repository_id,
                issue_id=issue.id,
                max_iterations=settings.max_iterations,
                require_plan_approval=False,
            )
            await session.commit()

        print(f"driving run {run.id} from {run.state}")
        final = await Orchestrator(runtime).advance(session, run)
        await session.commit()

        print(f"run reached {final}")

        if final is not RunState.WAITING_FOR_APPROVAL:
            await session.refresh(run)
            print(f"category: {run.failure_category}")
            print(f"detail:   {run.failure_detail}")
            raise SystemExit(1)

        diff = await review_service.latest_diff(session, run.id)
        print()
        print(
            f"diff: {diff.files_changed} file(s), +{diff.lines_added}/-{diff.lines_removed}, "
            f"risk {diff.risk_level} (score {diff.risk_score})"
        )
        print(diff.diff_text)
        print()

        # --- the gate, then the real push -------------------------------------------------
        approval = await review_service.approve(
            session, run=run, user_id=run.user_id, comment="approved by e2e_publish"
        )
        await session.commit()
        print(f"approved (diff_hash {approval.diff_hash[:12]}...)")

        github_client = await tokens.client_for_user(
            session, settings=settings, user_id=run.user_id
        )

        await create_pull_request(
            session, runtime=runtime, run=run, github_client=github_client
        )
        await session.commit()

        pull_request = (
            await session.execute(select(PullRequest).where(PullRequest.run_id == run.id))
        ).scalar_one()

        await session.refresh(run)

        print()
        print(f"PULL REQUEST: {pull_request.html_url}")
        print(f"branch:       {pull_request.branch}")
        print(f"run state:    {run.state}")


if __name__ == "__main__":
    # No WindowsSelectorEventLoopPolicy here, deliberately. The selector loop on Windows raises
    # NotImplementedError from subprocess_exec, and this script shells out to git and pytest.
    # The default proactor loop supports subprocesses; it just logs a cosmetic "Event loop is
    # closed" from asyncpg on exit, which is the better trade.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

