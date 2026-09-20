"""Agent run state machine.

The set of legal transitions is declared once, here, and enforced on every change.
An illegal transition raises instead of silently corrupting a run, which matters
because run state is the thing the UI, the resume logic and the approval gate all
trust.
"""

from enum import StrEnum


class RunState(StrEnum):
    CREATED = "CREATED"
    CLONING_REPOSITORY = "CLONING_REPOSITORY"
    ANALYZING_ISSUE = "ANALYZING_ISSUE"
    EXPLORING_REPOSITORY = "EXPLORING_REPOSITORY"
    INDEXING_REPOSITORY = "INDEXING_REPOSITORY"
    RETRIEVING_CONTEXT = "RETRIEVING_CONTEXT"
    PLANNING = "PLANNING"
    WAITING_FOR_PLAN_REVIEW = "WAITING_FOR_PLAN_REVIEW"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    ANALYZING_FAILURE = "ANALYZING_FAILURE"
    REVISING = "REVISING"
    VERIFYING = "VERIFYING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    PR_CREATED = "PR_CREATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FailureCategory(StrEnum):
    REPOSITORY_UNDERSTANDING_FAILURE = "REPOSITORY_UNDERSTANDING_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    PLANNING_FAILURE = "PLANNING_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    RECOVERY_FAILURE = "RECOVERY_FAILURE"
    SANDBOX_FAILURE = "SANDBOX_FAILURE"

    #: The model or embedding provider refused on quota. Distinct from a generic tool failure
    #: because the fix is "wait, or raise the limit", not "look for a bug". Retrying inside the
    #: run cannot help when a daily quota is exhausted.
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"

    TIMEOUT = "TIMEOUT"
    SECURITY_BLOCK = "SECURITY_BLOCK"
    HUMAN_REJECTION = "HUMAN_REJECTION"


#: States from which no further transition is possible.
TERMINAL_STATES: frozenset[RunState] = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}
)

#: States where the run is parked waiting on a human decision.
HUMAN_GATE_STATES: frozenset[RunState] = frozenset(
    {RunState.WAITING_FOR_PLAN_REVIEW, RunState.WAITING_FOR_APPROVAL}
)

#: Happy-path plus recovery transitions. FAILED and CANCELLED are handled separately
#: because they are reachable from any non-terminal state.
_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.CLONING_REPOSITORY}),
    RunState.CLONING_REPOSITORY: frozenset({RunState.ANALYZING_ISSUE}),
    RunState.ANALYZING_ISSUE: frozenset({RunState.EXPLORING_REPOSITORY}),
    RunState.EXPLORING_REPOSITORY: frozenset({RunState.INDEXING_REPOSITORY}),
    RunState.INDEXING_REPOSITORY: frozenset({RunState.RETRIEVING_CONTEXT}),
    RunState.RETRIEVING_CONTEXT: frozenset({RunState.PLANNING}),
    RunState.PLANNING: frozenset({RunState.WAITING_FOR_PLAN_REVIEW, RunState.IMPLEMENTING}),
    RunState.WAITING_FOR_PLAN_REVIEW: frozenset({RunState.IMPLEMENTING, RunState.PLANNING}),
    RunState.IMPLEMENTING: frozenset({RunState.TESTING}),
    RunState.TESTING: frozenset({RunState.VERIFYING, RunState.ANALYZING_FAILURE}),
    RunState.ANALYZING_FAILURE: frozenset({RunState.RETRIEVING_CONTEXT, RunState.REVISING}),
    RunState.REVISING: frozenset({RunState.TESTING}),
    RunState.VERIFYING: frozenset({RunState.WAITING_FOR_APPROVAL, RunState.ANALYZING_FAILURE}),
    RunState.WAITING_FOR_APPROVAL: frozenset({RunState.PR_CREATED, RunState.PLANNING}),
    RunState.PR_CREATED: frozenset({RunState.COMPLETED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class IllegalTransitionError(RuntimeError):
    def __init__(self, current: RunState, target: RunState) -> None:
        super().__init__(f"Illegal run state transition: {current} -> {target}")
        self.current = current
        self.target = target


def allowed_targets(current: RunState) -> frozenset[RunState]:
    """Every state reachable from ``current``, including failure and cancellation."""
    if current in TERMINAL_STATES:
        return frozenset()

    return _TRANSITIONS[current] | {RunState.FAILED, RunState.CANCELLED}


def can_transition(current: RunState, target: RunState) -> bool:
    return target in allowed_targets(current)


def assert_transition(current: RunState, target: RunState) -> None:
    if not can_transition(current, target):
        raise IllegalTransitionError(current, target)


def is_terminal(state: RunState) -> bool:
    return state in TERMINAL_STATES


def awaits_human(state: RunState) -> bool:
    return state in HUMAN_GATE_STATES
