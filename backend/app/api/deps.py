"""Shared FastAPI dependencies.

Dependencies are used for per-route concerns that produce a value (session, current
user, GitHub client). Middleware is used for cross-cutting concerns that apply to every
request (request id, logging, CORS).
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.db.session import get_session_factory
from app.github import tokens
from app.github.client import GitHubClient
from app.models.core import User
from app.security.crypto import build_session_codec

logger = logging.getLogger(__name__)


async def get_db() -> AsyncIterator[AsyncSession]:
    """One session per request, committed on success and rolled back on failure.

    Keeping the commit here means a handler that raises never leaves a half-written
    transaction behind.
    """
    factory = get_session_factory()

    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    settings: SettingsDep,
    session: DbDep,
    ase_session: Annotated[str | None, Cookie()] = None,
) -> User:
    if not ase_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    codec = build_session_codec(settings)

    try:
        user_id = codec.verify(ase_session)
    except Exception as exc:  # invalid signature, expired, malformed
        logger.info("rejected session cookie: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session"
        ) from exc

    user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_github_client(
    settings: SettingsDep, session: DbDep, user: CurrentUser
) -> GitHubClient:
    try:
        return await tokens.client_for_user(session, settings=settings, user_id=user.id)
    except tokens.ConnectionMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except tokens.CredentialUnreadableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


GitHubDep = Annotated[GitHubClient, Depends(get_github_client)]
