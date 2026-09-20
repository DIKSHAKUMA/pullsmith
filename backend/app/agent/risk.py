"""Risk scoring for a change set.

**This is a heuristic that directs a reviewer's attention. It is not a security guarantee and
is not a correctness check.** It answers "how carefully should a human look at this?", not "is
this safe?". The label says so, and the UI repeats it.

Scoring is deliberately rule-based rather than model-based. A reviewer needs to know *why*
something was flagged, and a number a language model invented is not auditable. Every point
added here comes with a stated reason.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.agent.diff import ChangeSet
from app.agent.testing import TestOutcome, all_passed

logger = logging.getLogger(__name__)


class RiskLevel(StrEnum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"


@dataclass
class RiskAssessment:
    level: RiskLevel
    score: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    #: Set when something is serious enough that the run should not reach approval at all.
    blocking: bool = False

    def describe(self) -> str:
        return f"{self.level} risk (score {self.score}): " + "; ".join(self.reasons or ["none"])


#: Patterns suggesting a credential was written into the diff. Deliberately broad: a false
#: positive costs a reviewer ten seconds, a missed one leaks a secret into a public PR.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(r"\bAQ\.[A-Za-z0-9\-_]{20,}"), "Google API key"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{20,}"), "Google API key"),
    (re.compile(r"sk-[A-Za-z0-9\-_]{16,}"), "provider API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (
        re.compile(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^:\s]+:[^@\s]+@"),
        "database URL with credentials",
    ),
)

#: Path patterns that change the blast radius of a merge.
_SECURITY_PATHS: tuple[tuple[str, str], ...] = (
    (".github/workflows/", "CI/CD workflow"),
    ("dockerfile", "container build"),
    ("docker-compose", "container orchestration"),
    ("alembic/versions/", "database migration"),
    ("migrations/", "database migration"),
    ("auth", "authentication code"),
    ("security", "security code"),
    ("permission", "authorisation code"),
    ("middleware", "request middleware"),
)

_DEPENDENCY_FILES: tuple[str, ...] = (
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "cargo.toml",
    "gemfile",
)

LOW_THRESHOLD = 3
MEDIUM_THRESHOLD = 7


def scan_for_secrets(diff_text: str) -> list[str]:
    """Finds credential-shaped strings in **added** lines only.

    Only added lines are scanned: a secret already present in the repository is a pre-existing
    problem, not something this change introduced, and flagging it would train reviewers to
    ignore the warning.
    """
    added = "\n".join(
        line[1:]
        for line in diff_text.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    )

    return sorted({label for pattern, label in _SECRET_PATTERNS if pattern.search(added)})


def assess(
    change_set: ChangeSet,
    test_outcomes: list[TestOutcome],
    *,
    planned_files: list[str] | None = None,
    iterations_used: int = 1,
) -> RiskAssessment:
    """Scores a change set for reviewer attention."""
    score = 0
    reasons: list[str] = []
    warnings: list[str] = []
    blocking = False

    # --- credentials: the only condition that blocks outright -------------------------
    leaked = scan_for_secrets(change_set.unified_diff())

    if leaked:
        score += 20
        blocking = True
        reasons.append("possible credential in the diff")
        warnings.append(
            f"The diff appears to add {', '.join(leaked)}. This must be removed before merging."
        )

    # --- tests -------------------------------------------------------------------------
    if not test_outcomes:
        score += 6
        reasons.append("no tests were run")
        warnings.append(
            "No test command was detected, so this change is unverified. Review manually."
        )
    elif not all_passed(test_outcomes):
        score += 8
        reasons.append("tests did not pass")

        if any(outcome.runner_error for outcome in test_outcomes):
            warnings.append("The test runner failed to execute, so nothing was verified.")
        else:
            failing = [name for outcome in test_outcomes for name in outcome.failing_tests]
            warnings.append(f"Failing tests: {', '.join(failing[:5]) or 'see output'}")

    # --- size --------------------------------------------------------------------------
    if change_set.file_count > 10:
        score += 3
        reasons.append(f"{change_set.file_count} files changed")
    elif change_set.file_count > 4:
        score += 1
        reasons.append(f"{change_set.file_count} files changed")

    total_lines = change_set.lines_added + change_set.lines_removed

    if total_lines > 400:
        # Weighted above the LOW threshold on its own: a change this large cannot be
        # reviewed at a glance, whatever else is true about it.
        score += 4
        reasons.append(f"{total_lines} lines changed")
        warnings.append("Large diff: check for unrelated refactoring.")
    elif total_lines > 150:
        score += 1
        reasons.append(f"{total_lines} lines changed")

    # --- sensitive areas ---------------------------------------------------------------
    for file_diff in change_set.files:
        lowered = file_diff.path.lower()

        for marker, label in _SECURITY_PATHS:
            if marker in lowered:
                score += 3
                reasons.append(f"touches {label}")
                warnings.append(f"{file_diff.path} is {label}; review the effect of merging it.")
                break

        if any(lowered.endswith(name) for name in _DEPENDENCY_FILES):
            score += 2
            reasons.append("dependency change")
            warnings.append(f"{file_diff.path} changes dependencies.")

    # --- scope creep -------------------------------------------------------------------
    if planned_files:
        planned = {path.lower() for path in planned_files}
        unplanned = [
            file_diff.path
            for file_diff in change_set.files
            if file_diff.path.lower() not in planned
        ]

        if unplanned:
            score += 2 * min(len(unplanned), 3)
            reasons.append(f"{len(unplanned)} file(s) not in the plan")
            warnings.append(
                f"Changed without being planned: {', '.join(unplanned[:5])}. "
                f"This is how scope creep enters a review."
            )

    # --- how hard it was ---------------------------------------------------------------
    if iterations_used >= 4:
        score += 2
        reasons.append(f"took {iterations_used} attempts")
        warnings.append(
            "The fix required several attempts, which often means the root cause was unclear."
        )

    if change_set.truncated:
        warnings.append("The diff was truncated for display; review the full changes in the PR.")

    level = (
        RiskLevel.low
        if score <= LOW_THRESHOLD
        else RiskLevel.medium
        if score <= MEDIUM_THRESHOLD
        else RiskLevel.high
    )

    if not reasons:
        reasons.append("small change, tests passed, no sensitive paths")

    logger.info("risk assessed: %s (score %s, blocking=%s)", level, score, blocking)

    return RiskAssessment(
        level=level, score=score, reasons=reasons, warnings=warnings, blocking=blocking
    )
