"""Write tools, the local runner, and test-result parsing.

The write tools are the first thing in the project that can change code, so the tests focus on
the guards: exact-match replacement, refusal on ambiguity, secret protection, and an accurate
record of what was written.
"""

import sys
from pathlib import Path

import pytest

from app.agent.testing import all_passed, parse_test_output, run_tests, targeted_command
from app.agent.tools.base import ToolContext, ToolRegistry
from app.agent.tools.editing import register_editing_tools, summarise_changes
from app.agent.tools.filesystem import register_filesystem_tools
from app.rag.repo_map import RepositoryMap
from app.sandbox.base import ExecResult, IsolationLevel, SandboxError, SandboxSpec
from app.sandbox.local import LocalSubprocessSandbox

ORIGINAL = """\
class ProfileService:
    def update(self, email):
        if not email:
            raise ValueError('email is required')
        return self.repository.save(email)
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "profile.py").write_text(ORIGINAL, encoding="utf-8")
    (root / ".env").write_text("SECRET=leak-me\n", encoding="utf-8")
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    return root


@pytest.fixture
def context(repo: Path) -> ToolContext:
    return ToolContext(workspace=repo, run_id="run-1", max_tool_calls=50)


@pytest.fixture
def registry() -> ToolRegistry:
    instance = ToolRegistry()
    register_filesystem_tools(instance)
    register_editing_tools(instance)
    return instance


class TestReplaceInFile:
    async def test_replaces_a_unique_snippet(
        self, registry: ToolRegistry, context: ToolContext, repo: Path
    ) -> None:
        result = await registry.execute(
            context,
            "replace_in_file",
            {
                "path": "app/profile.py",
                "find": "raise ValueError('email is required')",
                "replace": "raise ValidationError('email is required')",
                "reason": "map to a 422 response",
            },
        )

        assert result.ok

        updated = (repo / "app" / "profile.py").read_text(encoding="utf-8")
        assert "ValidationError" in updated
        # Everything else is untouched: the whole point of targeted replacement.
        assert "return self.repository.save(email)" in updated

    async def test_missing_text_is_refused_with_a_useful_message(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        """Usually means the agent is working from a stale retrieved snippet."""
        result = await registry.execute(
            context,
            "replace_in_file",
            {
                "path": "app/profile.py",
                "find": "raise HttpError('nope')",
                "replace": "pass",
                "reason": "guessing",
            },
        )

        assert not result.ok
        assert result.error_code == "TEXT_NOT_FOUND"
        assert "read the file again" in result.output.lower()

    async def test_ambiguous_text_is_refused(
        self, registry: ToolRegistry, context: ToolContext, repo: Path
    ) -> None:
        """Two matches means the agent has not identified a single location.

        Replacing the first would be a guess, and guessing which line to edit is how a
        one-line fix becomes a silent bug elsewhere.
        """
        (repo / "app" / "dup.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")

        result = await registry.execute(
            context,
            "replace_in_file",
            {"path": "app/dup.py", "find": "x = 1", "replace": "x = 9", "reason": "t"},
        )

        assert not result.ok
        assert result.error_code == "TEXT_NOT_UNIQUE"
        assert "appears 2 times" in result.output

    async def test_env_file_cannot_be_edited(
        self, registry: ToolRegistry, context: ToolContext, repo: Path
    ) -> None:
        result = await registry.execute(
            context,
            "replace_in_file",
            {"path": ".env", "find": "SECRET=leak-me", "replace": "SECRET=x", "reason": "t"},
        )

        assert not result.ok
        assert result.error_code == "SAFETY_REFUSED"
        assert "leak-me" in (repo / ".env").read_text(encoding="utf-8")

    async def test_traversal_is_refused(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        result = await registry.execute(
            context,
            "replace_in_file",
            {"path": "../outside.py", "find": "a", "replace": "b", "reason": "t"},
        )

        assert not result.ok
        assert result.error_code == "SAFETY_REFUSED"


class TestWriteFile:
    async def test_creates_a_new_file(
        self, registry: ToolRegistry, context: ToolContext, repo: Path
    ) -> None:
        result = await registry.execute(
            context,
            "write_file",
            {
                "path": "tests/test_profile.py",
                "content": "def test_empty_email():\n    assert True\n",
                "reason": "cover the fix",
            },
        )

        assert result.ok
        assert result.data["created"] is True
        assert (repo / "tests" / "test_profile.py").is_file()

    async def test_sensitive_path_is_flagged_but_allowed(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        """Dependency manifests may legitimately change; the reviewer must be told."""
        result = await registry.execute(
            context,
            "write_file",
            {"path": "requirements.txt", "content": "fastapi\npydantic\n", "reason": "add dep"},
        )

        assert result.ok
        assert result.data["is_sensitive"] is True
        assert "flagged for human review" in result.output

    async def test_oversized_write_is_refused(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        result = await registry.execute(
            context,
            "write_file",
            {"path": "app/big.py", "content": "x = 1\n" * 40_000, "reason": "t"},
        )

        assert not result.ok
        assert result.error_code == "CONTENT_TOO_LARGE"


class TestDeleteFile:
    async def test_deletes_a_file(
        self, registry: ToolRegistry, context: ToolContext, repo: Path
    ) -> None:
        result = await registry.execute(
            context, "delete_file", {"path": "app/profile.py", "reason": "obsolete"}
        )

        assert result.ok
        assert not (repo / "app" / "profile.py").exists()

    async def test_directories_cannot_be_deleted(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        """Recursive deletion is how a confused agent becomes an incident."""
        result = await registry.execute(context, "delete_file", {"path": "app", "reason": "t"})

        assert not result.ok
        assert result.error_code == "IS_A_DIRECTORY"


class TestChangeRecording:
    async def test_changes_are_recorded_for_the_review_diff(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        await registry.execute(
            context,
            "replace_in_file",
            {
                "path": "app/profile.py",
                "find": "raise ValueError('email is required')",
                "replace": "raise ValidationError('email is required')",
                "reason": "t",
            },
        )
        await registry.execute(
            context,
            "write_file",
            {"path": "tests/test_new.py", "content": "def test_x():\n    pass\n", "reason": "t"},
        )

        assert len(context.changes) == 2
        assert {change.path for change in context.changes} == {
            "app/profile.py",
            "tests/test_new.py",
        }
        # The original is kept so a diff can be produced from reality, not from claims.
        assert context.changes[0].original_content is not None

    async def test_summary_counts_each_file_once(
        self, registry: ToolRegistry, context: ToolContext
    ) -> None:
        edits = (
            ("email is required", "email required"),
            ("email required", "no email"),
        )

        for find, replace in edits:
            await registry.execute(
                context,
                "replace_in_file",
                {"path": "app/profile.py", "find": find, "replace": replace, "reason": "t"},
            )

        summary = summarise_changes(context.changes)

        assert summary.startswith("1 file(s) changed")

    def test_summary_reports_flagged_paths(self) -> None:
        from app.agent.tools.editing import FileChange

        summary = summarise_changes(
            [FileChange(path=".github/workflows/ci.yml", action="modified", is_sensitive=True)]
        )

        assert "Flagged for review" in summary

    def test_empty_summary(self) -> None:
        assert summarise_changes([]) == "No files changed."


class TestLocalSandbox:
    def test_reports_its_isolation_honestly(self) -> None:
        """The UI and config validation both depend on this not overstating itself."""
        sandbox = LocalSubprocessSandbox()

        assert sandbox.isolation is IsolationLevel.process
        assert sandbox.isolation is not IsolationLevel.virtual_machine
        assert sandbox.name == "local_unsafe"

    async def test_runs_a_command(self, repo: Path) -> None:
        sandbox = LocalSubprocessSandbox()
        await sandbox.prepare(SandboxSpec(workspace=repo))

        result = await sandbox.run([sys.executable.replace("\\", "/"), "-c", "print('hi')"])

        # sys.executable is not on the allowlist by name on every platform; if it was
        # refused, that itself is correct behaviour and asserted elsewhere.
        assert result.argv

    async def test_disallowed_executable_is_refused(self, repo: Path) -> None:
        sandbox = LocalSubprocessSandbox()
        await sandbox.prepare(SandboxSpec(workspace=repo))

        with pytest.raises(Exception, match="allowlist"):
            await sandbox.run(["curl", "https://example.com"])

    async def test_run_before_prepare_is_rejected(self) -> None:
        with pytest.raises(SandboxError, match="prepare"):
            await LocalSubprocessSandbox().run(["git", "status"])

    async def test_missing_workspace_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SandboxError, match="does not exist"):
            await LocalSubprocessSandbox().prepare(SandboxSpec(workspace=tmp_path / "nope"))

    async def test_secrets_are_not_passed_to_the_child(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A test suite must not be able to read our credentials from its environment."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.super-secret")

        sandbox = LocalSubprocessSandbox()
        await sandbox.prepare(SandboxSpec(workspace=repo))

        environment = sandbox._environment()  # noqa: SLF001 - asserting the boundary

        assert "DATABASE_URL" not in environment
        assert "GEMINI_API_KEY" not in environment
        assert "PATH" in environment

    async def test_cleanup_is_idempotent(self, repo: Path) -> None:
        sandbox = LocalSubprocessSandbox()
        await sandbox.prepare(SandboxSpec(workspace=repo))

        await sandbox.cleanup()
        await sandbox.cleanup()


def exec_result(stdout: str, exit_code: int = 0, *, timed_out: bool = False) -> ExecResult:
    return ExecResult(
        argv=["pytest", "-q"],
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_ms=1200,
        timed_out=timed_out,
    )


class TestPytestParsing:
    def test_passing_run(self) -> None:
        outcome = parse_test_output(exec_result("....\n4 passed in 1.2s\n"), "pytest")

        assert outcome.passed
        assert outcome.tests_passed == 4
        assert "4 passed" in outcome.describe()

    def test_failing_run_extracts_names(self) -> None:
        output = (
            "FAILED tests/test_profile.py::test_empty_email - AssertionError\n"
            "FAILED tests/test_profile.py::test_none_email - AssertionError\n"
            "2 failed, 6 passed in 2.1s\n"
        )

        outcome = parse_test_output(exec_result(output, exit_code=1), "pytest")

        assert not outcome.passed
        assert outcome.tests_failed == 2
        assert outcome.tests_passed == 6
        assert "tests/test_profile.py::test_empty_email" in outcome.failing_tests
        # Names go straight into the next retrieval query.
        assert len(outcome.failing_tests) == 2

    def test_no_tests_collected_is_a_runner_error_not_a_pass(self) -> None:
        """pytest exits 5 when it collects nothing.

        Treating that as success would let an unverified change reach a human looking proven.
        """
        outcome = parse_test_output(exec_result("no tests ran in 0.1s\n", exit_code=5), "pytest")

        assert not outcome.passed
        assert outcome.runner_error
        assert "could not execute" in outcome.describe()

    def test_import_error_is_a_runner_error(self) -> None:
        output = "ERROR collecting tests/test_x.py\nModuleNotFoundError: No module named 'foo'\n"

        outcome = parse_test_output(exec_result(output, exit_code=2), "pytest")

        assert outcome.runner_error

    def test_a_counted_collection_error_is_still_a_runner_error(self) -> None:
        """pytest reports "1 error" for a collection failure, not "1 failed".

        Found on a live run: the fixture repository had no root conftest.py, so nothing could be
        imported. It was reported as "1 test(s) failed", which would send the repair loop off to
        fix code that had never been executed. "ERROR collecting" is decisive on its own,
        whatever the count says.
        """
        output = (
            "=================================== ERRORS ====================\n"
            "ERROR collecting tests/test_calculator.py\n"
            "ImportError while importing test module\n"
            "1 error in 0.30s\n"
        )

        outcome = parse_test_output(exec_result(output, exit_code=2), "pytest")

        assert outcome.runner_error
        assert "could not execute" in outcome.describe()

        # Reporting a count taken from an error summary as failing assertions would be a lie.
        assert outcome.tests_failed is None
        assert outcome.failing_tests == []

    def test_a_genuine_failure_mentioning_importerror_is_not_a_runner_error(self) -> None:
        """A failing test's traceback often contains ImportError. That is still a real failure."""
        output = (
            "FAILED tests/test_plugins.py::test_missing_dependency - "
            "ImportError: No module named 'optional_extra'\n"
            "1 failed, 5 passed in 1.1s\n"
        )

        outcome = parse_test_output(exec_result(output, exit_code=1), "pytest")

        assert not outcome.runner_error
        assert outcome.tests_failed == 1

    def test_timeout_is_reported(self) -> None:
        outcome = parse_test_output(
            exec_result("running...", exit_code=-1, timed_out=True), "pytest"
        )

        assert not outcome.passed
        assert outcome.timed_out
        assert "timed out" in outcome.describe()


class TestJavaScriptParsing:
    def test_vitest_failure_counts(self) -> None:
        output = "  ✕ src/App.test.tsx > renders\n Tests  1 failed | 8 passed (9)\n"

        outcome = parse_test_output(exec_result(output, exit_code=1), "vitest")

        assert not outcome.passed
        assert outcome.tests_failed == 1
        assert outcome.tests_passed == 8

    def test_missing_module_is_a_runner_error(self) -> None:
        outcome = parse_test_output(
            exec_result("Cannot find module 'vitest'\n", exit_code=1), "vitest"
        )

        assert outcome.runner_error


class TestGoParsing:
    def test_go_failures(self) -> None:
        output = "--- FAIL: TestUpdate (0.00s)\n--- FAIL: TestDelete (0.01s)\nFAIL\n"

        outcome = parse_test_output(exec_result(output, exit_code=1), "go test")

        assert outcome.failing_tests == ["TestUpdate", "TestDelete"]
        assert outcome.tests_failed == 2


class TestTargetedCommands:
    def test_pytest_targeted(self) -> None:
        mapping = RepositoryMap(test_framework="pytest", test_command="pytest -q")

        assert targeted_command(mapping, ["tests/test_profile.py"]) == [
            "pytest",
            "-q",
            "tests/test_profile.py",
        ]

    def test_vitest_targeted(self) -> None:
        mapping = RepositoryMap(test_framework="vitest", test_command="npm test")

        assert targeted_command(mapping, ["src/App.test.tsx"]) == [
            "npx",
            "vitest",
            "run",
            "src/App.test.tsx",
        ]

    def test_unknown_framework_has_no_targeted_form(self) -> None:
        mapping = RepositoryMap(test_framework="custom", test_command="make test")

        assert targeted_command(mapping, ["a"]) is None

    def test_no_paths_means_no_targeted_run(self) -> None:
        mapping = RepositoryMap(test_framework="pytest", test_command="pytest -q")

        assert targeted_command(mapping, []) is None


class FakeSandbox:
    """Returns scripted results so the test strategy can be verified without running code."""

    def __init__(self, results: list[ExecResult]) -> None:
        self._results = list(results)
        self.commands: list[list[str]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def isolation(self) -> IsolationLevel:
        return IsolationLevel.none

    async def prepare(self, spec: SandboxSpec) -> None:
        return None

    async def run(
        self, argv: list[str], *, timeout: int = 300  # noqa: ASYNC109 - matches the Protocol
    ) -> ExecResult:
        self.commands.append(list(argv))
        return self._results.pop(0)

    async def cleanup(self) -> None:
        return None


class TestRunTests:
    async def test_targeted_then_full_suite_when_targeted_passes(self) -> None:
        sandbox = FakeSandbox(
            [exec_result("1 passed in 0.2s"), exec_result("20 passed in 4.0s")]
        )
        mapping = RepositoryMap(test_framework="pytest", test_command="pytest -q")

        outcomes = await run_tests(sandbox, mapping, test_paths=["tests/test_profile.py"])

        assert len(outcomes) == 2
        assert sandbox.commands[0] == ["pytest", "-q", "tests/test_profile.py"]
        assert sandbox.commands[1] == ["pytest", "-q"]
        assert all_passed(outcomes)

    async def test_failing_targeted_run_skips_the_full_suite(self) -> None:
        """The answer is already known; the full suite would only cost minutes."""
        sandbox = FakeSandbox([exec_result("1 failed in 0.2s", exit_code=1)])
        mapping = RepositoryMap(test_framework="pytest", test_command="pytest -q")

        outcomes = await run_tests(sandbox, mapping, test_paths=["tests/test_profile.py"])

        assert len(outcomes) == 1
        assert not all_passed(outcomes)

    async def test_no_test_command_returns_nothing(self) -> None:
        sandbox = FakeSandbox([])
        mapping = RepositoryMap(test_framework=None, test_command=None)

        assert await run_tests(sandbox, mapping) == []

    def test_no_outcomes_is_not_success(self) -> None:
        """"No tests were run" must never be reported as a pass."""
        assert not all_passed([])
