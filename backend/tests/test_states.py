"""State machine legality tests.

These matter because the approval gate, the resume logic and the UI all trust the run
state. A silent illegal transition would let a run reach PR creation without approval.
"""

import pytest

from app.agent.states import (
    TERMINAL_STATES,
    IllegalTransitionError,
    RunState,
    allowed_targets,
    assert_transition,
    awaits_human,
    can_transition,
    is_terminal,
)


def test_happy_path_is_legal() -> None:
    path = [
        RunState.CREATED,
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
        RunState.PR_CREATED,
        RunState.COMPLETED,
    ]

    for current, target in zip(path, path[1:], strict=False):
        assert can_transition(current, target), f"{current} -> {target} should be legal"


def test_repair_loop_is_legal() -> None:
    assert can_transition(RunState.TESTING, RunState.ANALYZING_FAILURE)
    assert can_transition(RunState.ANALYZING_FAILURE, RunState.RETRIEVING_CONTEXT)
    assert can_transition(RunState.ANALYZING_FAILURE, RunState.REVISING)
    assert can_transition(RunState.REVISING, RunState.TESTING)


def test_cannot_skip_straight_to_pr() -> None:
    # The whole safety model depends on this being impossible.
    assert not can_transition(RunState.CREATED, RunState.PR_CREATED)
    assert not can_transition(RunState.TESTING, RunState.PR_CREATED)
    assert not can_transition(RunState.VERIFYING, RunState.PR_CREATED)


def test_pr_requires_waiting_for_approval() -> None:
    approvers = [
        state for state in RunState if can_transition(state, RunState.PR_CREATED)
    ]
    assert approvers == [RunState.WAITING_FOR_APPROVAL]


def test_failure_and_cancel_reachable_from_any_active_state() -> None:
    for state in RunState:
        if state in TERMINAL_STATES:
            continue
        assert can_transition(state, RunState.FAILED)
        assert can_transition(state, RunState.CANCELLED)


def test_terminal_states_are_dead_ends() -> None:
    for state in TERMINAL_STATES:
        assert is_terminal(state)
        assert allowed_targets(state) == frozenset()


def test_illegal_transition_raises() -> None:
    with pytest.raises(IllegalTransitionError):
        assert_transition(RunState.CREATED, RunState.COMPLETED)


def test_human_gates_identified() -> None:
    assert awaits_human(RunState.WAITING_FOR_APPROVAL)
    assert awaits_human(RunState.WAITING_FOR_PLAN_REVIEW)
    assert not awaits_human(RunState.TESTING)
