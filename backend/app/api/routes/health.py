"""Liveness and configuration visibility."""

import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbDep, SettingsDep
from app.config.settings import AppEnv
from app.schemas.api import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(session: DbDep, settings: SettingsDep) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        database = "connected"
    except Exception as exc:
        logger.warning("health check database probe failed: %s", type(exc).__name__)
        database = "unreachable"

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
    )
