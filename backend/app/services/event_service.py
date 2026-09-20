"""Run events and guarded state transitions.

Two invariants live here:

1. A state change and the event describing it are written in the *same* transaction.
   If one is rolled back, so is the other, so the timeline can never disagree with the
   run's actual state.
2. Event sequences are allocated from ``agent_run.last_event_sequence`` under a row
   lock, so they are gap-free and strictly ordered. That ordering is what lets the SSE
   endpoint resume from ``Last-Event-ID`` without replaying or dropping events.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.states import FailureCategory, RunState, assert_transition, is_terminal
from app.db.base import new_id, utc_now
from app.models.core import AgentEvent, AgentRun

logger = logging.getLogger(__name__)


class EventKind:
    STATE_CHANGED = "state_changed"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    TEST_RUN = "test_run"
    APPROVAL = "approval"


async def _load_run_for_update(session: AsyncSession, run_id: str) -> AgentRun:
    stmt = select(AgentRun).where(AgentRun.id == run_id)

    # SQLite has no row locks; the test suite runs single-writer so ordering holds.
    if session.bind is not None and session.bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()

    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise LookupError(f"Run not found: {run_id}")
    return run


async def append_event(
    session: AsyncSession,
    run_id: str,
    *,
    kind: str,
    message: str,
    level: str = "info",
    state: RunState | None = None,
    payload: dict[str, Any] | None = None,
) -> AgentEvent:
    """Appends one event, allocating the next sequence number atomically."""
    run = await _load_run_for_update(session, run_id)
    return await _append_event_for(session, run, kind=kind, message=message, level=level,
                                   state=state, payload=payload)


async def _append_event_for(
    session: AsyncSession,
    run: AgentRun,
    *,
    kind: str,
    message: str,
    level: str,
    state: RunState | None,
    payload: dict[str, Any] | None,
) -> AgentEvent:
    run.last_event_sequence += 1

    event = AgentEvent(
        id=new_id(),
        run_id=run.id,
        sequence=run.last_event_sequence,
        kind=kind,
        state=str(state) if state is not None else run.state,
        message=message,
        level=level,
        payload=payload,
        created_at=utc_now(),
    )
    session.add(event)
    await session.flush()
    return event


async def transition(
    session: AsyncSession,
    run_id: str,
    target: RunState,
    *,
    message: str | None = None,
    failure_category: FailureCategory | None = None,
    failure_detail: str | None = None,
    payload: dict[str, Any] | None = None,
) -> AgentRun:
    """Moves a run to ``target``, rejecting illegal transitions.

    Raises ``IllegalTransitionError`` before touching anything, so a buggy caller
    cannot leave the run in an inconsistent state.
    """
    run = await _load_run_for_update(session, run_id)
    current = RunState(run.state)

    assert_transition(current, target)

    run.state = str(target)

    if target is RunState.CLONING_REPOSITORY and run.started_at is None:
        run.started_at = utc_now()

    if is_terminal(target):
        run.finished_at = utc_now()

    if failure_category is not None:
        run.failure_category = str(failure_category)
    if failure_detail is not None:
        run.failure_detail = failure_detail

    await _append_event_for(
        session,
        run,
        kind=EventKind.STATE_CHANGED,
        message=message or f"{current} -> {target}",
        level="error" if target is RunState.FAILED else "info",
        state=target,
        payload=payload,
    )

    logger.info("run %s transitioned %s -> %s", run.id, current, target)
    return run


async def list_events(
    session: AsyncSession,
    run_id: str,
    *,
    after_sequence: int = 0,
    limit: int = 500,
) -> list[AgentEvent]:
    result = await session.execute(
        select(AgentEvent)
        .where(AgentEvent.run_id == run_id, AgentEvent.sequence > after_sequence)
        .order_by(AgentEvent.sequence)
        .limit(limit)
    )
    return list(result.scalars())
