"""Running the repository's tests, and reading the result.

This is the only *objective* signal in the whole system. Everything else — the plan, the root
cause, the model's confidence — is opinion. The test suite either passes or it does not.

Two deliberate choices:

**Targeted tests first.** If the plan names test files, run those before the full suite. On a
large repository the full suite can take many minutes, and the repair loop may run several
times, so a fast first signal matters more than completeness. The full suite runs afterwards
to catch regressions the targeted run would miss.

**Parse the output, do not trust the exit code alone.** A non-zero exit says something went
wrong; it does not say whether two tests failed or the runner could not start. Those need
different responses, so failure names and counts are extracted.
"""

import logging
import re
from dataclasses import dataclass, field

from app.rag.repo_map import RepositoryMap
from app.sandbox.base import ExecResult, SandboxRunner

logger = logging.getLogger(__name__)

TEST_TIMEOUT_SECONDS = 600
TARGETED_TIMEOUT_SECONDS = 240


@dataclass
class TestOutcome:
    #: Stops pytest trying to collect this as a test class because of its name.
    __test__ = False

    command: list[str]
    passed: bool
    exit_code: int
    duration_ms: int
    timed_out: bool = False

    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_skipped: int | None = None

    failing_tests: list[str] = field(default_factory=list)
    output_summary: str = ""

    #: True when the runner itself failed - missing dependency, collection error, no tests
    #: found. A different problem from a failing assertion, needing a different fix.
    runner_error: bool = False

    def describe(self) -> str:
        if self.timed_out:
            return f"Tests timed out after {self.duration_ms // 1000}s"

        if self.runner_error:
            return f"Test runner could not execute (exit {self.exit_code})"

        if self.passed:
            counted = f"{self.tests_passed} passed" if self.tests_passed else "suite passed"
            return f"Tests passed ({counted})"

        failed = self.tests_failed if self.tests_failed is not None else "some"
        names = ", ".join(self.failing_tests[:5])
        suffix = f": {names}" if names else ""

        return f"{failed} test(s) failed{suffix}"


#: pytest: "5 passed, 2 failed, 1 skipped in 3.2s"
_PYTEST_COUNTS = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors)")
_PYTEST_FAILURE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
#: Decisive on their own: these can only mean the runner never got as far as running tests.
#: Counted errors are irrelevant here — pytest reports "1 error" for a collection failure, and
#: treating that as "1 test failed" sends the repair loop off to fix code that was never run.
_PYTEST_RUNNER_BROKEN = re.compile(
    r"(ERROR collecting|INTERNALERROR|no tests ran|error: unrecognized arguments)",
    re.IGNORECASE,
)

#: Suggestive only: these strings also appear in the traceback of a genuinely failing test, so
#: they count as a runner problem only when nothing actually failed.
_PYTEST_COLLECT_ERROR = re.compile(
    r"(ModuleNotFoundError|ImportError)",
    re.IGNORECASE,
)

#: vitest / jest: "Tests  3 failed | 8 passed (11)"
_JS_FAILED = re.compile(r"(?:Tests?|✕)\s+.*?(\d+)\s+failed", re.IGNORECASE)
_JS_PASSED = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_JS_FAILURE_NAME = re.compile(r"^\s*(?:✕|×|FAIL)\s+(.+)$", re.MULTILINE)
_JS_RUNNER_ERROR = re.compile(
    r"(Cannot find module|No test files found|command not found|ERR_MODULE_NOT_FOUND)",
    re.IGNORECASE,
)

#: go test: "--- FAIL: TestUpdate (0.00s)"
_GO_FAILURE = re.compile(r"^--- FAIL:\s+(\S+)", re.MULTILINE)


def parse_test_output(result: ExecResult, framework: str | None) -> TestOutcome:
    """Turns raw runner output into structured facts."""
    combined = f"{result.stdout}\n{result.stderr}"

    outcome = TestOutcome(
        command=result.argv,
        passed=result.ok,
        exit_code=result.exit_code,
        duration_ms=result.duration_ms,
        timed_out=result.timed_out,
        output_summary=result.summary(),
    )

    if result.timed_out:
        outcome.passed = False
        return outcome

    if framework == "pytest":
        counts = {name: int(value) for value, name in _PYTEST_COUNTS.findall(combined)}
        outcome.tests_passed = counts.get("passed")
        outcome.tests_failed = counts.get("failed") or counts.get("error") or counts.get("errors")
        outcome.tests_skipped = counts.get("skipped")
        outcome.failing_tests = _PYTEST_FAILURE.findall(combined)[:20]

        # Exit code 5 is pytest's "no tests collected", which is a setup problem rather than
        # a code problem and must not be reported as a passing suite.
        if (
            result.exit_code == 5
            or (not result.ok and _PYTEST_RUNNER_BROKEN.search(combined))
            or (
                not result.ok
                and not outcome.tests_failed
                and _PYTEST_COLLECT_ERROR.search(combined)
            )
        ):
            outcome.runner_error = True

            # The count came from an error summary, not from assertions, so reporting it as a
            # failing-test count would be a lie.
            outcome.tests_failed = None
            outcome.failing_tests = []

    elif framework in {"vitest", "jest", "playwright"}:
        failed = _JS_FAILED.search(combined)
        passed = _JS_PASSED.search(combined)

        outcome.tests_failed = int(failed.group(1)) if failed else None
        outcome.tests_passed = int(passed.group(1)) if passed else None
        outcome.failing_tests = [name.strip() for name in _JS_FAILURE_NAME.findall(combined)][:20]

        if not result.ok and _JS_RUNNER_ERROR.search(combined):
            outcome.runner_error = True

    elif framework == "go test":
        outcome.failing_tests = _GO_FAILURE.findall(combined)[:20]
        outcome.tests_failed = len(outcome.failing_tests) or None

    elif not result.ok and not outcome.failing_tests:
        # Unknown framework: the exit code is all we have, so say so rather than guessing.
        outcome.runner_error = "not found" in combined.lower()

    outcome.passed = result.ok and not outcome.runner_error

    return outcome


def targeted_command(mapping: RepositoryMap, test_paths: list[str]) -> list[str] | None:
    """Builds a command that runs only the named tests.

    Returns None when the framework does not support it or nothing was named, in which case
    the caller falls back to the full suite.
    """
    if not test_paths or not mapping.test_command:
        return None

    framework = mapping.test_framework

    if framework == "pytest":
        return ["pytest", "-q", *test_paths]

    if framework == "vitest":
        return ["npx", "vitest", "run", *test_paths]

    if framework == "jest":
        return ["npx", "jest", "--ci", *test_paths]

    if framework == "go test":
        return ["go", "test", *test_paths]

    return None


async def run_tests(
    sandbox: SandboxRunner,
    mapping: RepositoryMap,
    *,
    test_paths: list[str] | None = None,
) -> list[TestOutcome]:
    """Runs the repository's tests, targeted first then the full suite.

    Returns every attempt so the caller can see both the fast signal and the regression
    check. An empty list means no test command could be determined at all.
    """
    if not mapping.test_command:
        logger.warning("no test command detected; cannot verify objectively")
        return []

    outcomes: list[TestOutcome] = []

    targeted = targeted_command(mapping, test_paths or [])

    if targeted:
        logger.info("running targeted tests: %s", " ".join(targeted))
        result = await sandbox.run(targeted, timeout=TARGETED_TIMEOUT_SECONDS)
        outcome = parse_test_output(result, mapping.test_framework)
        outcomes.append(outcome)

        # A failing targeted run is the answer already; running the full suite would only
        # add minutes and noise before the agent revises.
        if not outcome.passed:
            return outcomes

    full = mapping.test_command.split()
    logger.info("running full suite: %s", " ".join(full))

    result = await sandbox.run(full, timeout=TEST_TIMEOUT_SECONDS)
    outcomes.append(parse_test_output(result, mapping.test_framework))

    return outcomes


def all_passed(outcomes: list[TestOutcome]) -> bool:
    """True only when tests actually ran and every attempt passed.

    An empty list is False on purpose: "no tests were run" must never be reported as
    success, which would let an unverified change reach a human as if it were proven.
    """
    return bool(outcomes) and all(outcome.passed for outcome in outcomes)
