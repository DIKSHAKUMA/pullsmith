"""End-to-end orchestrator runs against a real git repository on disk.

What is real here: the git clone, the file scan, the repository map, tree-sitter chunking, the
SQLite writes, the tool registry and every safety check, the diff builder, the risk scorer, the
state machine and the approval gate.

What is substituted, and why:

* the **model**, by a scripted provider — a real one is slow, costs money and is not
  reproducible, so a failing test would not tell you whether your code or the model changed;
* the **sandbox**, by a runner returning canned test output — the point is to exercise how the
  orchestrator reacts to a pass or a failure, not to install a Python environment per test;
* **retrieval**, because similarity search needs Postgres. Retrieval has its own tests against
  a real instance; here it is a stub so the state machine can be tested on SQLite.

The scripts are exact: the fake provider raises if the code asks for one more turn than was
written, so an accidental extra model call fails the test rather than passing quietly.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import Orchestrator
from app.agent.runtime import AgentRuntime, build_registry
from app.agent.states import RunState
from app.config.settings import Settings
from app.db.base import new_id
from app.llm.base import QuotaExceededError
from app.llm.fake import FakeLLMProvider, ScriptedTurn, text_turn, tool_turn
from app.models.core import AgentRun, Repository
from app.rag.embeddings import FakeEmbeddingProvider
from app.sandbox.base import ExecResult, IsolationLevel, SandboxSpec
from app.services import event_service, review_service

# --------------------------------------------------------------------------- fixture repo

BROKEN_SOURCE = '''\
"""A tiny module with a deliberate bug."""


def divide(numerator, denominator):
    return numerator / denominator
'''

FIXED_SOURCE = '''\
"""A tiny module with a deliberate bug."""


def divide(numerator, denominator):
    if denominator == 0:
        raise ValueError("denominator must not be zero")

    return numerator / denominator
'''

TEST_SOURCE = '''\
from calc import divide


def test_divide():
    assert divide(6, 3) == 2
'''


#: 0xC0000142, Windows STATUS_DLL_INIT_FAILED. On a memory-constrained machine a subprocess
#: can fail to start at all, which is an environment problem and not a test failure.
_DLL_INIT_FAILED = 3221225794


def _git(*args: str, cwd: Path) -> None:
    # Fixed argument list, fixed executable, test-only paths from tmp_path.
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=False,
        capture_output=True,
    )

    if result.returncode == _DLL_INIT_FAILED:
        pytest.skip(
            f"git {args[0]} could not start (0xC0000142): the machine is out of memory "
            f"for a new process. Re-run this file on its own."
        )

    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:400]}"
        )


@pytest.fixture
def origin_repo(tmp_path: Path) -> Path:
    """A real git repository the agent can clone, with a real bug in it."""
    root = tmp_path / "origin"
    (root / "tests").mkdir(parents=True)

    (root / "calc.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    (root / "tests" / "test_calc.py").write_text(TEST_SOURCE, encoding="utf-8")
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    _git("init", "-b", "main", cwd=root)
    _git("config", "user.email", "agent@example.test", cwd=root)
    _git("config", "user.name", "Agent Test", cwd=root)
    _git("add", ".", cwd=root)
    _git("commit", "-m", "initial", cwd=root)

    return root


# ------------------------------------------------------------------------- fake sandbox


@dataclass
class ScriptedSandbox:
    """A sandbox that returns canned command output.

    Reports ``IsolationLevel.none`` because it is not isolating anything. The honesty matters:
    a test double that claimed isolation would make the UI's isolation badge untestable.
    """

    results: list[ExecResult] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    prepared: list[Path] = field(default_factory=list)
    cleanups: int = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def isolation(self) -> IsolationLevel:
        return IsolationLevel.none

    async def prepare(self, spec: SandboxSpec) -> None:
        self.prepared.append(spec.workspace)

    async def run(
        self,
        argv: list[str],
        *,
        timeout: int = 300,  # noqa: ASYNC109 - matches the SandboxRunner protocol
    ) -> ExecResult:
        self.commands.append(argv)

        if self.results:
            result = self.results.pop(0)
            return ExecResult(
                argv=argv,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=result.duration_ms,
                timed_out=result.timed_out,
            )

        return ExecResult(
            argv=argv, exit_code=0, stdout="1 passed in 0.01s", stderr="", duration_ms=5
        )

    async def cleanup(self) -> None:
        self.cleanups += 1


def passing_tests() -> ExecResult:
    return ExecResult(argv=[], exit_code=0, stdout="1 passed in 0.02s", stderr="", duration_ms=20)


def failing_tests() -> ExecResult:
    return ExecResult(
        argv=[],
        exit_code=1,
        stdout="FAILED tests/test_calc.py::test_divide_by_zero\n1 failed, 1 passed in 0.03s",
        stderr="",
        duration_ms=30,
    )


# ------------------------------------------------------------------------- scripted model


def analysis_turn() -> ScriptedTurn:
    return text_turn(
        json.dumps(
            {
                "problem_summary": "divide() raises ZeroDivisionError instead of ValueError",
                "expected_behaviour": "A clear ValueError when the denominator is zero",
                "actual_behaviour": "ZeroDivisionError propagates to the caller",
                "referenced_symbols": ["divide"],
                "referenced_paths": ["calc.py"],
                "search_queries": ["divide by zero handling"],
                "is_actionable": True,
            }
        )
    )


def unactionable_turn() -> ScriptedTurn:
    return text_turn(
        json.dumps(
            {
                "problem_summary": "The issue does not say what is wrong",
                "expected_behaviour": "unknown",
                "actual_behaviour": "unknown",
                "search_queries": ["unknown"],
                "is_actionable": False,
                "clarification_needed": "Which endpoint, and what did you see?",
            }
        )
    )


def plan_turn() -> ScriptedTurn:
    return text_turn(
        json.dumps(
            {
                "problem_understanding": "divide() does not guard against a zero denominator",
                "relevant_files": ["calc.py"],
                "suspected_root_cause": "No validation before the division",
                "root_cause_confidence": "high",
                "proposed_changes": [
                    {"path": "calc.py", "intent": "Raise ValueError when denominator is zero"}
                ],
                "tests_to_add_or_update": ["tests/test_calc.py"],
                "verification_strategy": "Run pytest on tests/test_calc.py",
            }
        )
    )


def verification_turn() -> ScriptedTurn:
    return text_turn(
        json.dumps(
            {
                "issue_addressed": True,
                "tests_pass": True,
                "unrelated_changes_detected": False,
                "summary": "Added a zero check to divide()",
                "confidence": "high",
            }
        )
    )


def edit_turn() -> ScriptedTurn:
    return tool_turn(
        "replace_in_file",
        {
            "path": "calc.py",
            "find": "    return numerator / denominator",
            "replace": (
                '    if denominator == 0:\n'
                '        raise ValueError("denominator must not be zero")\n\n'
                "    return numerator / denominator"
            ),
            "reason": "Guard against a zero denominator",
        },
    )


def happy_path_script() -> list[ScriptedTurn]:
    """Exactly the turns one clean run needs, in order."""
    return [
        analysis_turn(),                        # _analyse_issue
        text_turn("I have read enough."),       # _plan: tool loop, no tools requested
        plan_turn(),                            # _plan: conclude_with_schema
        edit_turn(),                            # _implement: one edit
        text_turn("Added the guard."),          # _implement: loop ends
        text_turn("The diff matches the plan."),  # _assess: verify tool loop
        verification_turn(),                    # _assess: conclude_with_schema
    ]


# ------------------------------------------------------------------------------- wiring


@pytest.fixture
def runtime(settings: Settings, tmp_path: Path):  # noqa: ANN201
    """Builds an AgentRuntime with fakes, and a retriever that returns nothing.

    Returning nothing is the honest stub: retrieval needs Postgres, and pretending to have
    found relevant code would test a fiction. The orchestrator is expected to warn and carry
    on, which is behaviour worth exercising anyway.
    """

    async def retrieve(_queries: list[str], _symbols: list[str]) -> list:
        return []

    def make(sandbox: ScriptedSandbox, turns: list[ScriptedTurn]) -> AgentRuntime:
        return AgentRuntime(
            settings=settings,
            llm=FakeLLMProvider(turns),
            embeddings=FakeEmbeddingProvider(dimensions=settings.embedding_dimensions),
            sandbox=sandbox,
            registry=build_registry(),
            workspace_root=tmp_path / "workspaces",
            retriever_factory=lambda _snapshot_id: retrieve,
        )

    return make


async def _transition_states(db: AsyncSession, run_id: str) -> list[str]:
    """States from state-change events only.

    Informational events inherit the run's current state, so counting every event would
    report each state several times.
    """
    return [
        event.state
        for event in await event_service.list_events(db, run_id)
        if event.kind == event_service.EventKind.STATE_CHANGED and event.state
    ]


async def _make_run(
    db: AsyncSession, seeded: dict[str, str], origin: Path, **kwargs
) -> AgentRun:  # noqa: ANN003
    repository = await db.get(Repository, seeded["repository_id"])
    repository.clone_url = str(origin)

    run = AgentRun(
        id=new_id(),
        user_id=seeded["user_id"],
        repository_id=seeded["repository_id"],
        issue_id=seeded["issue_id"],
        state=str(kwargs.pop("state", RunState.CREATED)),
        max_iterations=kwargs.pop("max_iterations", 3),
        require_plan_approval=kwargs.pop("require_plan_approval", False),
        **kwargs,
    )
    db.add(run)
    await db.flush()
    await db.commit()
    return run


# -------------------------------------------------------------------------------- tests


async def test_full_run_reaches_the_approval_gate(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """One clean pass: clone, analyse, index, plan, edit, test, diff, park for a human."""
    sandbox = ScriptedSandbox(results=[passing_tests(), passing_tests()])
    agent_runtime = runtime(sandbox, happy_path_script())
    run = await _make_run(db, seeded, origin_repo)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.WAITING_FOR_APPROVAL

    # The edit really happened, in the real checkout.
    checkout = agent_runtime.workspace_for(run.id)
    assert checkout.is_dir(), "the workspace must survive so the pull request can be built"
    assert "denominator must not be zero" in (checkout / "calc.py").read_text(encoding="utf-8")

    # And it is visible to a reviewer as a diff, with a risk score.
    diff = await review_service.latest_diff(db, run.id)
    assert diff is not None
    assert diff.files_changed == 1
    assert "calc.py" in diff.diff_text
    assert diff.risk_blocking is False

    plan = await review_service.latest_plan(db, run.id)
    assert plan is not None
    assert plan.root_cause_confidence == "high"

    # Nothing was skipped: the timeline records every state it passed through.
    states = await _transition_states(db, run.id)

    for expected in (
        RunState.CLONING_REPOSITORY,
        RunState.ANALYZING_ISSUE,
        RunState.EXPLORING_REPOSITORY,
        RunState.INDEXING_REPOSITORY,
        RunState.RETRIEVING_CONTEXT,
        RunState.PLANNING,
        RunState.IMPLEMENTING,
        RunState.TESTING,
        RunState.VERIFYING,
        RunState.WAITING_FOR_APPROVAL,
    ):
        assert str(expected) in states, f"{expected} missing from the timeline"

    assert agent_runtime.llm.turns_remaining == 0, "the script was not consumed exactly"


async def test_plan_gate_parks_before_any_edit(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """With plan approval required, the run stops before writing anything."""
    sandbox = ScriptedSandbox()
    agent_runtime = runtime(
        sandbox, [analysis_turn(), text_turn("Read enough."), plan_turn()]
    )
    run = await _make_run(db, seeded, origin_repo, require_plan_approval=True)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.WAITING_FOR_PLAN_REVIEW
    assert await review_service.latest_plan(db, run.id) is not None
    assert await review_service.latest_diff(db, run.id) is None

    checkout = agent_runtime.workspace_for(run.id)
    assert (checkout / "calc.py").read_text(encoding="utf-8") == BROKEN_SOURCE
    assert sandbox.commands == [], "no tests should run before a plan is approved"


async def test_resumes_from_the_plan_gate_and_recovers_the_plan(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """A second worker picks the run up at IMPLEMENTING with nothing in memory.

    The plan comes back from the database, the repository map is rebuilt from the checkout, and
    the issue analysis is re-derived. It must not replay the clone or the indexing.
    """
    first = runtime(ScriptedSandbox(), [analysis_turn(), text_turn("Read enough."), plan_turn()])
    run = await _make_run(db, seeded, origin_repo, require_plan_approval=True)

    assert await Orchestrator(first).advance(db, run) is RunState.WAITING_FOR_PLAN_REVIEW

    # A human approves the plan.
    await event_service.transition(db, run.id, RunState.IMPLEMENTING, message="Plan approved")
    await db.commit()

    second = runtime(
        ScriptedSandbox(results=[passing_tests(), passing_tests()]),
        [
            analysis_turn(),  # re-derived, because it was never persisted
            edit_turn(),
            text_turn("Added the guard."),
            text_turn("The diff matches the plan."),
            verification_turn(),
        ],
    )
    # The workspace from the first pass is reused; both runtimes share a root.
    assert second.workspace_for(run.id).is_dir()

    final = await Orchestrator(second).advance(db, run)

    assert final is RunState.WAITING_FOR_APPROVAL

    states = await _transition_states(db, run.id)
    assert states.count(str(RunState.CLONING_REPOSITORY)) == 1
    assert states.count(str(RunState.INDEXING_REPOSITORY)) == 1

    diff = await review_service.latest_diff(db, run.id)
    assert diff is not None and diff.files_changed == 1


async def test_recovers_edits_from_git_after_losing_them_to_a_crash(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """Resuming at TESTING rebuilds the change set from the git working tree.

    The write tools record original file contents in memory. A crash loses that, but the
    checkout is pinned to one commit, so git still holds the originals and the diff can be
    reconstructed truthfully rather than guessed at.
    """
    agent_runtime = runtime(ScriptedSandbox(), happy_path_script())
    run = await _make_run(db, seeded, origin_repo)

    # Drive the run far enough to have a checkout, an index and a stored plan.
    await Orchestrator(agent_runtime).advance(db, run)

    checkout = agent_runtime.workspace_for(run.id)
    assert checkout.is_dir()

    # Simulate the crash: the state says TESTING, and nothing is in memory.
    run.state = str(RunState.TESTING)
    await db.commit()

    resumed = runtime(
        ScriptedSandbox(results=[passing_tests(), passing_tests()]),
        [text_turn("The diff matches the plan."), verification_turn()],
    )

    final = await Orchestrator(resumed).advance(db, run)

    assert final is RunState.WAITING_FOR_APPROVAL

    diff = await review_service.latest_diff(db, run.id)
    assert diff is not None
    assert diff.files_changed == 1

    # Reconstructed against the committed version, so the guard shows as an addition and the
    # untouched line below it shows as context rather than a rewrite.
    assert '+        raise ValueError("denominator must not be zero")' in diff.diff_text
    assert "\n     return numerator / denominator" in diff.diff_text
    assert diff.lines_removed == 0


async def test_missing_workspace_after_edits_fails_rather_than_publishing(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """A lost workspace past the edit stage must fail, not re-clone and lose the change."""
    agent_runtime = runtime(ScriptedSandbox(), [])
    run = await _make_run(db, seeded, origin_repo, state=RunState.VERIFYING)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.FAILED
    await db.refresh(run)
    assert run.failure_category == "TOOL_FAILURE"
    assert "workspace" in (run.failure_detail or "").lower()


async def test_quota_exhaustion_is_reported_as_itself(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """Running out of provider quota is not a bug in the agent, and the run should say so.

    Reported generically it looks like a crash and costs an afternoon of debugging. This
    happened on the third live run.
    """
    agent_runtime = runtime(
        ScriptedSandbox(), [ScriptedTurn(error=QuotaExceededError("quota exceeded"))]
    )
    run = await _make_run(db, seeded, origin_repo)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.FAILED
    await db.refresh(run)
    assert run.failure_category == "QUOTA_EXHAUSTED"
    assert "quota" in (run.failure_detail or "").lower()


async def test_unactionable_issue_stops_instead_of_guessing(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    agent_runtime = runtime(ScriptedSandbox(), [unactionable_turn()])
    run = await _make_run(db, seeded, origin_repo)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.FAILED
    await db.refresh(run)
    assert run.failure_category == "PLANNING_FAILURE"
    assert "not actionable" in (run.failure_detail or "")
    assert await review_service.latest_diff(db, run.id) is None


async def test_failing_tests_end_the_run_as_a_test_failure(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    """Tests that never pass must not reach a human dressed up as a success.

    The diff is still stored, because a reviewer investigating the failure wants to see what
    was attempted. What does not happen is a transition to WAITING_FOR_APPROVAL.
    """
    sandbox = ScriptedSandbox(results=[failing_tests()] * 6)
    agent_runtime = runtime(
        sandbox,
        [
            analysis_turn(),
            text_turn("Read enough."),
            plan_turn(),
            # Attempt 1
            edit_turn(),
            text_turn("Added the guard."),
            # Failure analysis after attempt 1
            text_turn(
                json.dumps(
                    {
                        "error_type": "AssertionError",
                        "error_summary": "test_divide_by_zero still fails",
                        "likely_cause": "The guard raises the wrong exception type",
                        "caused_by_our_change": True,
                        "additional_search_queries": [],
                        "suggested_fix": "Raise ValueError, not TypeError",
                        "is_recoverable": False,
                    }
                )
            ),
        ],
    )
    run = await _make_run(db, seeded, origin_repo, max_iterations=3)

    final = await Orchestrator(agent_runtime).advance(db, run)

    assert final is RunState.FAILED
    await db.refresh(run)
    assert run.failure_category == "TEST_FAILURE"

    diff = await review_service.latest_diff(db, run.id)
    assert diff is not None, "the attempted change is still worth showing a reviewer"

    assert str(RunState.WAITING_FOR_APPROVAL) not in await _transition_states(db, run.id)


async def test_terminal_run_is_left_alone(
    db: AsyncSession,
    seeded: dict[str, str],
    origin_repo: Path,
    runtime,  # noqa: ANN001
) -> None:
    agent_runtime = runtime(ScriptedSandbox(), [])
    run = await _make_run(db, seeded, origin_repo, state=RunState.CANCELLED)

    assert await Orchestrator(agent_runtime).advance(db, run) is RunState.CANCELLED
    assert agent_runtime.llm.turns_remaining == 0
