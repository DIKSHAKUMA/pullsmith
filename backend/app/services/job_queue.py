"""Durable job queue backed by Postgres.

Why not Redis + Celery: run state already lives in Postgres, and on a 6 GB development
machine an extra broker process costs more than it earns. ``FOR UPDATE SKIP LOCKED``
gives durable at-least-once claiming with one table and no new infrastructure.

Delivery semantics are **at-least-once**. A worker can die after doing work but before
marking the job complete, so the lease expires and the job is retried. Every handler
must therefore be idempotent, which is why the orchestrator resumes from persisted run
state rather than assuming it starts from scratch.
"""

import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id, utc_now
from app.models.core import Job

logger = logging.getLogger(__name__)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobKind:
    EXECUTE_RUN = "execute_run"
    INDEX_REPOSITORY = "index_repository"

    #: Enqueued when a human approves a diff. Separate from EXECUTE_RUN so the push happens
    #: outside the reviewer's HTTP request and can be retried on a transient GitHub failure
    #: without re-running the agent.
    CREATE_PULL_REQUEST = "create_pull_request"


async def enqueue(
    session: AsyncSession,
    *,
    kind: str,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    delay_seconds: int = 0,
    max_attempts: int = 3,
) -> Job:
    job = Job(
        id=new_id(),
        kind=kind,
        run_id=run_id,
        payload=payload,
        status=JobStatus.QUEUED,
        max_attempts=max_attempts,
        run_after=utc_now() + timedelta(seconds=delay_seconds),
    )
    session.add(job)
    await session.flush()
    return job


async def claim_next(
    session: AsyncSession,
    *,
    worker_id: str,
    lease_seconds: int = 120,
) -> Job | None:
    """Claims one runnable job, or returns None.

    The lock is taken inside the same transaction as the status update, so two workers
    polling simultaneously cannot both receive the same row: the second one skips the
    locked row instead of blocking on it.
    """
    now = utc_now()

    stmt = (
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.run_after <= now)
        .order_by(Job.run_after)
        .limit(1)
    )

    if session.bind is not None and session.bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update(skip_locked=True)

    job = (await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        return None

    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.locked_at = now
    job.locked_by = worker_id
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)

    await session.flush()
    return job


async def heartbeat(session: AsyncSession, job_id: str, *, lease_seconds: int = 120) -> None:
    """Extends the lease so a long but healthy job is not reclaimed as dead."""
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(lease_expires_at=utc_now() + timedelta(seconds=lease_seconds))
    )


async def mark_succeeded(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status=JobStatus.SUCCEEDED, locked_by=None, lease_expires_at=None)
    )


async def mark_failed(
    session: AsyncSession,
    job_id: str,
    *,
    error: str,
    retry_delay_seconds: int = 30,
) -> str:
    """Fails a job, retrying it while attempts remain.

    Returns the resulting status so the caller can log or emit an event.
    """
    job = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()

    if job.attempts < job.max_attempts:
        job.status = JobStatus.QUEUED
        job.run_after = utc_now() + timedelta(seconds=retry_delay_seconds)
    else:
        job.status = JobStatus.FAILED

    job.last_error = error[:2000]
    job.locked_by = None
    job.lease_expires_at = None

    await session.flush()
    return job.status


async def reclaim_expired(session: AsyncSession) -> int:
    """Requeues jobs whose lease expired, which is how worker crashes recover."""
    result = await session.execute(
        update(Job)
        .where(Job.status == JobStatus.RUNNING, Job.lease_expires_at < utc_now())
        .values(status=JobStatus.QUEUED, locked_by=None, lease_expires_at=None)
    )
    count = result.rowcount or 0

    if count:
        logger.warning("reclaimed %s job(s) with expired leases", count)

    return count
