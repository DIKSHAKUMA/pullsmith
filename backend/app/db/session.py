"""Async engine and session management.

The engine is created once per process and shared. Sessions are per-request (or
per-job in the worker) and always closed, which is why they are handed out through a
context manager rather than a module-level object.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import Settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    kwargs: dict[str, object] = {"echo": False, "future": True}

    if not settings.is_sqlite:
        # Neon pools connections itself and scales compute to zero, so keep the local
        # pool small and recycle before idle connections are dropped upstream.
        kwargs.update(
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=280,
        )

    _engine = create_async_engine(settings.database_url, **kwargs)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Engine not initialised. Call init_engine() during startup.")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()

    _engine = None
    _session_factory = None
