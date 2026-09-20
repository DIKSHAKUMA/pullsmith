"""Event ordering, transition guarding, queue claiming and crash recovery."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.states import IllegalTransitionError, RunState
from app.db.base import new_id
from app.models.core import AgentRun, Job
from app.services import event_service, job_queue


async def _make_run(db: AsyncSession, seeded: dict[str, str]) -> AgentRun:
    run = AgentRun(
        id=new_id(),
        user_id=seeded["user_id"],
        repository_id=seeded["repository_id"],
        issue_id=seeded["issue_id"],
        state=str(RunState.CREATED),
        max_iterations=5,
    )
    db.add(run)
    await db.flush()
    return run


async def test_event_sequences_are_ordered_and_gap_free(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)

    for index in range(5):
        await event_service.append_event(
            db, run.id, kind=event_service.EventKind.INFO, message=f"step {index}"
        )

    events = await event_service.list_events(db, run.id)
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]


async def test_list_events_after_sequence_supports_resume(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)

    for index in range(4):
        await event_service.append_event(
            db, run.id, kind=event_service.EventKind.INFO, message=f"step {index}"
        )

    # Simulates a client reconnecting with Last-Event-ID: 2
    resumed = await event_service.list_events(db, run.id, after_sequence=2)
    assert [event.sequence for event in resumed] == [3, 4]


async def test_transition_writes_state_and_event_together(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)

    await event_service.transition(db, run.id, RunState.CLONING_REPOSITORY)

    refreshed = (
        await db.execute(select(AgentRun).where(AgentRun.id == run.id))
    ).scalar_one()

    assert refreshed.state == str(RunState.CLONING_REPOSITORY)
    assert refreshed.started_at is not None

    events = await event_service.list_events(db, run.id)
    assert events[-1].kind == event_service.EventKind.STATE_CHANGED
    assert events[-1].state == str(RunState.CLONING_REPOSITORY)


async def test_illegal_transition_is_rejected(db: AsyncSession, seeded: dict[str, str]) -> None:
    run = await _make_run(db, seeded)

    with pytest.raises(IllegalTransitionError):
        await event_service.transition(db, run.id, RunState.PR_CREATED)


async def test_terminal_transition_sets_finished_at(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)

    await event_service.transition(db, run.id, RunState.CANCELLED, message="cancelled")

    refreshed = (await db.execute(select(AgentRun).where(AgentRun.id == run.id))).scalar_one()
    assert refreshed.finished_at is not None


async def test_claim_marks_job_running_and_increments_attempts(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)
    await job_queue.enqueue(db, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id)

    claimed = await job_queue.claim_next(db, worker_id="worker-a")

    assert claimed is not None
    assert claimed.status == job_queue.JobStatus.RUNNING
    assert claimed.attempts == 1
    assert claimed.locked_by == "worker-a"
    assert claimed.lease_expires_at is not None


async def test_claimed_job_is_not_handed_out_twice(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)
    await job_queue.enqueue(db, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id)

    first = await job_queue.claim_next(db, worker_id="worker-a")
    second = await job_queue.claim_next(db, worker_id="worker-b")

    assert first is not None
    assert second is None


async def test_failed_job_retries_until_attempts_exhausted(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)
    job = await job_queue.enqueue(
        db, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id, max_attempts=2
    )

    await job_queue.claim_next(db, worker_id="worker-a")
    status = await job_queue.mark_failed(db, job.id, error="boom")
    assert status == job_queue.JobStatus.QUEUED

    # Second attempt exhausts the budget.
    job.run_after = job.created_at
    await db.flush()
    await job_queue.claim_next(db, worker_id="worker-a")
    status = await job_queue.mark_failed(db, job.id, error="boom again")
    assert status == job_queue.JobStatus.FAILED


async def test_expired_lease_is_reclaimed(db: AsyncSession, seeded: dict[str, str]) -> None:
    """This is the worker-crash recovery path."""
    run = await _make_run(db, seeded)
    job = await job_queue.enqueue(db, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id)

    await job_queue.claim_next(db, worker_id="doomed-worker", lease_seconds=-1)

    reclaimed = await job_queue.reclaim_expired(db)
    assert reclaimed == 1

    refreshed = (await db.execute(select(Job).where(Job.id == job.id))).scalar_one()
    assert refreshed.status == job_queue.JobStatus.QUEUED
    assert refreshed.locked_by is None


async def test_delayed_job_is_not_claimable_yet(
    db: AsyncSession, seeded: dict[str, str]
) -> None:
    run = await _make_run(db, seeded)
    await job_queue.enqueue(
        db, kind=job_queue.JobKind.EXECUTE_RUN, run_id=run.id, delay_seconds=300
    )

    assert await job_queue.claim_next(db, worker_id="worker-a") is None
