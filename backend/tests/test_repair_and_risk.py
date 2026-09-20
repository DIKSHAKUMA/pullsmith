"""Repair loop, diff generation and risk scoring.

The repair loop is the part that must not run away: these tests pin the bounds, and check that
a failure actually becomes the input to the next attempt rather than just being logged.
"""

import json
from pathlib import Path

import pytest

from app.agent.diff import build_change_set
from app.agent.repair import implement_and_repair, verify
from app.agent.risk import RiskLevel, assess, scan_for_secrets
from app.agent.schemas import ImplementationPlan
from app.agent.testing import TestOutcome
from app.agent.tools.base import ToolContext, ToolRegistry
from app.agent.tools.editing import FileChange, register_editing_tools
from app.agent.tools.filesystem import register_filesystem_tools
from app.llm.fake import FakeLLMProvider, text_turn, tool_turn
from app.rag.repo_map import RepositoryMap
from app.sandbox.base import ExecResult, IsolationLevel, SandboxSpec

ORIGINAL = """\
class ProfileService:
    def update(self, email):
        if not email:
            raise ValueError('email is required')
        return self.repository.save(email)
"""

PLAN_JSON = {
    "problem_understanding": "ValueError escapes as a 500",
    "relevant_files": ["app/profile.py"],
    "suspected_root_cause": "ValueError is never mapped to a 422",
    "root_cause_confidence": "medium",
    "proposed_changes": [
        {"path": "app/profile.py", "intent": "raise a domain error", "is_new_file": False}
    ],
    "tests_to_add_or_update": ["tests/test_profile.py"],
    "risks": [],
    "verification_strategy": "run pytest",
    "out_of_scope": [],
}

FAILURE_JSON = {
    "failing_tests": ["tests/test_profile.py::test_empty_email"],
    "error_type": "AssertionError",
    "error_summary": "expected 422, received 500",
    "likely_cause": "the exception is raised but not translated by the handler",
    "caused_by_our_change": True,
    "files_to_inspect": ["app/api/errors.py"],
    "additional_search_queries": ["exception handler mapping", "validation error response"],
    "suggested_fix": "register a handler for the domain error",
    "is_recoverable": True,
}

VERIFICATION_JSON = {
    "issue_addressed": True,
    "tests_pass": True,
    "unrelated_changes_detected": False,
    "concerns": [],
    "summary": "Empty email now returns a validation error",
    "confidence": "high",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "profile.py").write_text(ORIGINAL, encoding="utf-8")
    return root


@pytest.fixture
def context(repo: Path) -> ToolContext:
    return ToolContext(workspace=repo, run_id="run-1", max_tool_calls=60)


@pytest.fixture
def registry() -> ToolRegistry:
    instance = ToolRegistry()
    register_filesystem_tools(instance)
    register_editing_tools(instance)
    return instance


@pytest.fixture
def plan() -> ImplementationPlan:
    return ImplementationPlan.model_validate(PLAN_JSON)


@pytest.fixture
def mapping() -> RepositoryMap:
    return RepositoryMap(test_framework="pytest", test_command="pytest -q")


def result_for(stdout: str, exit_code: int = 0) -> ExecResult:
    return ExecResult(
        argv=["pytest", "-q"], exit_code=exit_code, stdout=stdout, stderr="", duration_ms=900
    )


class ScriptedSandbox:
    def __init__(self, results: list[ExecResult]) -> None:
        self._results = list(results)
        self.commands: list[list[str]] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def isolation(self) -> IsolationLevel:
        return IsolationLevel.none

    async def prepare(self, spec: SandboxSpec) -> None:
        return None

    async def run(
        self, argv: list[str], *, timeout: int = 300  # noqa: ASYNC109 - matches the Protocol
    ) -> ExecResult:
        self.commands.append(list(argv))
        return self._results.pop(0) if self._results else result_for("1 passed in 0.1s")

    async def cleanup(self) -> None:
        return None


EDIT_ARGS = {
    "path": "app/profile.py",
    "find": "raise ValueError('email is required')",
    "replace": "raise ValidationError('email is required')",
    "reason": "map to 422",
}


class TestFirstAttemptSucceeds:
    async def test_passing_tests_end_the_loop(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        provider = FakeLLMProvider(
            [tool_turn("replace_in_file", EDIT_ARGS), text_turn("done editing")]
        )
        sandbox = ScriptedSandbox([result_for("1 passed in 0.1s"), result_for("20 passed")])

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
        )

        assert result.succeeded
        assert result.iterations_used == 1
        assert result.stopped_because == "tests_passed"

    async def test_the_edit_actually_reached_the_file(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
        repo: Path,
    ) -> None:
        provider = FakeLLMProvider(
            [tool_turn("replace_in_file", EDIT_ARGS), text_turn("done")]
        )
        sandbox = ScriptedSandbox([result_for("1 passed"), result_for("20 passed")])

        await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
        )

        assert "ValidationError" in (repo / "app" / "profile.py").read_text(encoding="utf-8")


class TestRepairCycle:
    async def test_failure_leads_to_a_second_attempt_that_passes(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        provider = FakeLLMProvider(
            [
                tool_turn("replace_in_file", EDIT_ARGS),
                text_turn("first attempt done"),
                text_turn(json.dumps(FAILURE_JSON)),  # failure analysis
                tool_turn("read_file", {"path": "app/profile.py"}),
                text_turn("second attempt done"),
            ]
        )
        sandbox = ScriptedSandbox(
            [
                result_for("FAILED tests/test_profile.py::test_empty_email\n1 failed", 1),
                result_for("1 passed in 0.2s"),
                result_for("20 passed in 4s"),
            ]
        )

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
        )

        assert result.succeeded
        assert result.iterations_used == 2
        assert result.attempts[0].failure_analysis is not None
        assert result.attempts[0].failure_analysis.error_type == "AssertionError"

    async def test_failure_output_is_fed_into_the_revision_prompt(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        """The whole point of the loop: the failure becomes the next input."""
        provider = FakeLLMProvider(
            [
                tool_turn("replace_in_file", EDIT_ARGS),
                text_turn("done"),
                text_turn(json.dumps(FAILURE_JSON)),
                text_turn("revised"),
            ]
        )
        sandbox = ScriptedSandbox(
            [
                result_for("FAILED tests/test_profile.py::test_empty_email\n1 failed", 1),
                result_for("1 passed"),
                result_for("20 passed"),
            ]
        )

        await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
            max_iterations=3,
        )

        revision_prompt = provider.requests[-1].messages[1].content

        assert "test_empty_email" in revision_prompt
        assert "the exception is raised but not translated" in revision_prompt
        # Test output is third-party text, so it arrives as data.
        assert "<untrusted-data" in revision_prompt

    async def test_failing_tests_drive_new_retrieval(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        """Failure output names real symbols, which is what lexical search is best at."""
        captured: list[tuple[list[str], list[str]]] = []

        async def retriever(queries: list[str], symbols: list[str]):  # noqa: ANN202
            captured.append((queries, symbols))
            return []

        provider = FakeLLMProvider(
            [
                tool_turn("replace_in_file", EDIT_ARGS),
                text_turn("done"),
                text_turn(json.dumps(FAILURE_JSON)),
                text_turn("revised"),
            ]
        )
        sandbox = ScriptedSandbox(
            [
                result_for("FAILED tests/test_profile.py::test_empty_email\n1 failed", 1),
                result_for("1 passed"),
                result_for("20 passed"),
            ]
        )

        await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
            retriever=retriever,
            max_iterations=3,
        )

        assert captured
        queries, symbols = captured[0]
        assert "exception handler mapping" in queries
        assert "test_empty_email" in symbols


class TestBounds:
    async def test_max_iterations_stops_the_loop(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        """A model that never fixes it must not loop forever."""
        turns = []
        for _ in range(3):
            turns.append(text_turn("attempting"))
            turns.append(text_turn(json.dumps(FAILURE_JSON)))

        provider = FakeLLMProvider(turns)
        sandbox = ScriptedSandbox([result_for("1 failed", 1) for _ in range(6)])

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
            max_iterations=3,
        )

        assert not result.succeeded
        assert result.iterations_used == 3
        assert result.stopped_because == "max_iterations_reached"

    async def test_unrecoverable_analysis_stops_early(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        """An honest early stop beats four more identical failures."""
        unrecoverable = {
            **FAILURE_JSON,
            "is_recoverable": False,
            "likely_cause": "a required dependency is not installed in the environment",
        }

        provider = FakeLLMProvider(
            [text_turn("attempting"), text_turn(json.dumps(unrecoverable))]
        )
        sandbox = ScriptedSandbox([result_for("ModuleNotFoundError\n1 failed", 1)])

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
            max_iterations=5,
        )

        assert not result.succeeded
        assert result.stopped_because == "analysis_says_unrecoverable"
        assert result.iterations_used == 1

    async def test_missing_test_command_stops_immediately(
        self, registry: ToolRegistry, context: ToolContext, plan: ImplementationPlan
    ) -> None:
        """Without tests nothing can be verified, and iterating will not change that."""
        provider = FakeLLMProvider([text_turn("edited")])
        sandbox = ScriptedSandbox([])

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=RepositoryMap(test_framework=None, test_command=None),
            plan=plan,
            retrieved=[],
        )

        assert not result.succeeded
        assert result.stopped_because == "no_test_command"

    async def test_tool_budget_exhaustion_stops_the_loop(
        self, registry: ToolRegistry, repo: Path, plan: ImplementationPlan, mapping: RepositoryMap
    ) -> None:
        tight = ToolContext(workspace=repo, run_id="run-1", max_tool_calls=1)

        provider = FakeLLMProvider(
            [tool_turn("read_file", {"path": "app/profile.py"}), text_turn("done")]
        )
        sandbox = ScriptedSandbox([result_for("1 passed"), result_for("20 passed")])

        result = await implement_and_repair(
            provider=provider,
            registry=registry,
            context=tight,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
        )

        assert result.stopped_because == "tool_budget_exhausted"

    async def test_observer_sees_each_attempt(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        plan: ImplementationPlan,
        mapping: RepositoryMap,
    ) -> None:
        events: list[tuple[str, dict]] = []

        async def observer(kind: str, payload: dict) -> None:
            events.append((kind, payload))

        provider = FakeLLMProvider(
            [tool_turn("replace_in_file", EDIT_ARGS), text_turn("done")]
        )
        sandbox = ScriptedSandbox([result_for("1 passed"), result_for("20 passed")])

        await implement_and_repair(
            provider=provider,
            registry=registry,
            context=context,
            sandbox=sandbox,
            mapping=mapping,
            plan=plan,
            retrieved=[],
            observer=observer,
        )

        kinds = [kind for kind, _ in events]

        assert "test_run" in kinds
        test_event = next(payload for kind, payload in events if kind == "test_run")
        assert test_event["passed"] is True


class TestVerification:
    async def test_verification_is_read_only(
        self, registry: ToolRegistry, context: ToolContext, plan: ImplementationPlan
    ) -> None:
        """Verification must report problems, not quietly fix them."""
        provider = FakeLLMProvider(
            [
                tool_turn("write_file", {"path": "app/x.py", "content": "x", "reason": "sneak"}),
                text_turn("cannot do that"),
                text_turn(json.dumps(VERIFICATION_JSON)),
            ]
        )

        result, _usage = await verify(
            provider=provider,
            registry=registry,
            context=context,
            plan=plan,
            diff_text="--- a/app/profile.py\n+++ b/app/profile.py\n",
            outcomes=[],
        )

        assert result.issue_addressed
        advertised = {tool["name"] for tool in provider.requests[0].tools or []}
        assert "write_file" not in advertised

    async def test_verification_can_report_concerns(
        self, registry: ToolRegistry, context: ToolContext, plan: ImplementationPlan
    ) -> None:
        concerned = {
            **VERIFICATION_JSON,
            "unrelated_changes_detected": True,
            "concerns": ["Also reformatted an unrelated file"],
            "confidence": "low",
        }

        provider = FakeLLMProvider([text_turn("checking"), text_turn(json.dumps(concerned))])

        result, _usage = await verify(
            provider=provider,
            registry=registry,
            context=context,
            plan=plan,
            diff_text="diff",
            outcomes=[],
        )

        assert result.unrelated_changes_detected
        assert result.concerns


class TestDiffGeneration:
    def test_diff_is_built_from_disk_not_from_claims(self, repo: Path) -> None:
        (repo / "app" / "profile.py").write_text(
            ORIGINAL.replace("ValueError", "ValidationError"), encoding="utf-8"
        )

        change_set = build_change_set(
            repo,
            [FileChange(path="app/profile.py", action="modified", original_content=ORIGINAL)],
        )

        assert change_set.file_count == 1
        assert "-            raise ValueError" in change_set.unified_diff()
        assert "+            raise ValidationError" in change_set.unified_diff()
        assert change_set.lines_added == 1
        assert change_set.lines_removed == 1

    def test_repeated_edits_to_one_file_are_one_entry(self, repo: Path) -> None:
        """And the diff is against how the file looked before the agent touched it."""
        (repo / "app" / "profile.py").write_text("final content\n", encoding="utf-8")

        change_set = build_change_set(
            repo,
            [
                FileChange(path="app/profile.py", action="modified", original_content=ORIGINAL),
                FileChange(
                    path="app/profile.py", action="modified", original_content="intermediate\n"
                ),
            ],
        )

        assert change_set.file_count == 1
        assert "ProfileService" in change_set.unified_diff()

    def test_new_file_shows_as_all_additions(self, repo: Path) -> None:
        (repo / "tests").mkdir()
        (repo / "tests" / "test_new.py").write_text("def test_x():\n    pass\n", encoding="utf-8")

        change_set = build_change_set(
            repo, [FileChange(path="tests/test_new.py", action="created", original_content=None)]
        )

        assert change_set.lines_added == 2
        assert change_set.lines_removed == 0

    def test_unchanged_file_is_omitted(self, repo: Path) -> None:
        """An agent that writes a file back identically has not made a change."""
        change_set = build_change_set(
            repo,
            [FileChange(path="app/profile.py", action="modified", original_content=ORIGINAL)],
        )

        assert change_set.file_count == 0
        assert change_set.summary() == "No changes."

    def test_sensitive_path_is_marked(self, repo: Path) -> None:
        (repo / ".github" / "workflows").mkdir(parents=True)
        (repo / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")

        change_set = build_change_set(
            repo,
            [
                FileChange(
                    path=".github/workflows/ci.yml", action="created", original_content=None
                )
            ],
        )

        assert change_set.sensitive_paths == [".github/workflows/ci.yml"]


def passing() -> TestOutcome:
    return TestOutcome(
        command=["pytest"], passed=True, exit_code=0, duration_ms=100, tests_passed=10
    )


def failing() -> TestOutcome:
    return TestOutcome(
        command=["pytest"],
        passed=False,
        exit_code=1,
        duration_ms=100,
        tests_failed=2,
        failing_tests=["tests/test_a.py::test_b"],
    )


def change_set_with(path: str, added: int = 5, removed: int = 2):  # noqa: ANN201
    from app.agent.diff import ChangeSet, FileDiff

    return ChangeSet(
        files=[
            FileDiff(
                path=path,
                action="modified",
                lines_added=added,
                lines_removed=removed,
                is_sensitive=False,
                diff_text=f"--- a/{path}\n+++ b/{path}\n+added line\n",
            )
        ]
    )


class TestRiskScoring:
    def test_small_passing_change_is_low_risk(self) -> None:
        assessment = assess(change_set_with("app/profile.py"), [passing()])

        assert assessment.level is RiskLevel.low
        assert not assessment.blocking

    def test_failing_tests_raise_the_level(self) -> None:
        assessment = assess(change_set_with("app/profile.py"), [failing()])

        assert assessment.level is not RiskLevel.low
        assert "tests did not pass" in assessment.reasons

    def test_no_tests_run_is_penalised(self) -> None:
        """Unverified must never look the same as verified."""
        assessment = assess(change_set_with("app/profile.py"), [])

        assert "no tests were run" in assessment.reasons
        assert any("unverified" in warning for warning in assessment.warnings)

    def test_ci_workflow_change_is_flagged(self) -> None:
        assessment = assess(change_set_with(".github/workflows/deploy.yml"), [passing()])

        assert any("CI/CD" in reason for reason in assessment.reasons)

    def test_auth_change_is_flagged(self) -> None:
        assessment = assess(change_set_with("app/auth/session.py"), [passing()])

        assert any("authentication" in reason for reason in assessment.reasons)

    def test_dependency_change_is_flagged(self) -> None:
        assessment = assess(change_set_with("requirements.txt"), [passing()])

        assert "dependency change" in assessment.reasons

    def test_unplanned_files_are_flagged_as_scope_creep(self) -> None:
        assessment = assess(
            change_set_with("app/unrelated.py"),
            [passing()],
            planned_files=["app/profile.py"],
        )

        assert any("not in the plan" in reason for reason in assessment.reasons)
        assert any("scope creep" in warning for warning in assessment.warnings)

    def test_many_iterations_add_risk(self) -> None:
        assessment = assess(
            change_set_with("app/profile.py"), [passing()], iterations_used=5
        )

        assert any("attempts" in reason for reason in assessment.reasons)

    def test_large_diff_adds_risk(self) -> None:
        assessment = assess(change_set_with("app/profile.py", added=500, removed=300), [passing()])

        assert assessment.level is not RiskLevel.low
        assert any("lines changed" in reason for reason in assessment.reasons)


class TestSecretScanning:
    @pytest.mark.parametrize(
        "secret",
        [
            # All synthetic. An earlier version used a fragment of a real Gemini key here.
            "ghp_abcdefghijklmnopqrstuvwxyz1234",
            "AQ.Fake000ExampleKeyForTestsOnly000",
            "sk-abcdefghijklmnopqrstuvwx",
            "postgresql://user:password@host/db",
        ],
    )
    def test_secrets_in_added_lines_block_the_run(self, secret: str) -> None:
        from app.agent.diff import ChangeSet, FileDiff

        change_set = ChangeSet(
            files=[
                FileDiff(
                    path="app/config.py",
                    action="modified",
                    lines_added=1,
                    lines_removed=0,
                    is_sensitive=False,
                    diff_text=f"--- a/app/config.py\n+++ b/app/config.py\n+KEY = '{secret}'\n",
                )
            ]
        )

        assessment = assess(change_set, [passing()])

        assert assessment.blocking
        assert assessment.level is RiskLevel.high

    def test_pre_existing_secret_is_not_blamed_on_this_change(self) -> None:
        """Only added lines are scanned.

        Flagging a secret that was already in the repository would train reviewers to ignore
        the warning.
        """
        diff = "--- a/app/config.py\n+++ b/app/config.py\n-KEY = 'ghp_abcdefghijklmnopqrst1234'\n"

        assert scan_for_secrets(diff) == []

    def test_ordinary_code_is_not_flagged(self) -> None:
        diff = "--- a/app/x.py\n+++ b/app/x.py\n+def compute(total):\n+    return total * 2\n"

        assert scan_for_secrets(diff) == []
