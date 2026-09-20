"""Liveness and configuration visibility."""

import logging

from fastapi import APIRouter
from sqlalchemy import func, select, text

from app.api.deps import DbDep, SettingsDep
from app.config.settings import AppEnv
from app.db.base import utc_now
from app.models.core import Job
from app.schemas.api import HealthResponse
from app.services import job_queue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


async def _queue_state(session: DbDep) -> tuple[int, int | None]:
    """Queue depth and the age of the oldest waiting job.

    The API cannot see the worker directly, and adding a heartbeat table for it would be a
    migration for one boolean. The queue already carries the answer: work piling up unclaimed
    means nothing is consuming it.
    """
    try:
        result = await session.execute(
            select(func.count(), func.min(Job.run_after)).where(
                Job.status == job_queue.JobStatus.QUEUED
            )
        )
        count, oldest = result.one()
    except Exception as exc:  # noqa: BLE001 - health must not fail because of this
        logger.warning("queue probe failed: %s", type(exc).__name__)
        return 0, None

    if not count or oldest is None:
        return int(count or 0), None

    # SQLite hands back a naive datetime, Postgres an aware one. Compare like with like.
    now = utc_now() if oldest.tzinfo else utc_now().replace(tzinfo=None)

    return int(count), max(0, int((now - oldest).total_seconds()))


@router.get("/health", response_model=HealthResponse)
async def health(session: DbDep, settings: SettingsDep) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        database = "connected"
    except Exception as exc:
        logger.warning("health check database probe failed: %s", type(exc).__name__)
        database = "unreachable"

    queued_jobs, oldest_queued_seconds = await _queue_state(session)

    return HealthResponse(
        status="ok",
        database=database,
        app_env=str(settings.app_env),
        sandbox_backend=str(settings.sandbox_backend),
        github_oauth_configured=bool(
            settings.github_client_id and settings.github_client_secret
        ),
        # Mirrors the guard on the route itself rather than restating the rule, so the two
        # cannot disagree about whether token sign-in is permitted.
        dev_login_available=settings.app_env is not AppEnv.production,
        queued_jobs=queued_jobs,
        oldest_queued_job_seconds=oldest_queued_seconds,
    )
