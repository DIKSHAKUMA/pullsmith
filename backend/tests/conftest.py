"""Test fixtures.

Tests run against SQLite in-memory. The Phase 1 schema deliberately uses portable
column types so the whole suite runs without a network round trip to Neon, which keeps
the feedback loop fast on a two-core machine.

Phase 2 adds pgvector columns, which are Postgres-only. Those tests will be marked and
run against a real Postgres instance.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.db.session as session_module
from app.config.settings import AppEnv, Settings
from app.db.base import Base, new_id

# Importing every model module registers all tables on Base.metadata, so create_all in the
# engine fixture builds the full schema rather than only what a test happens to import.
from app.models import rag, review  # noqa: F401
from app.models.core import GitHubConnection, Issue, Repository, User
from app.security.crypto import build_cipher, build_session_codec


@pytest.fixture(scope="session")
def event_loop():  # noqa: ANN201
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    # _env_file=None isolates the suite from the developer's local .env so results do
    # not depend on whose machine it runs on.
    return Settings(
        _env_file=None,
        app_env=AppEnv.test,
        database_url="sqlite+aiosqlite:///:memory:",
        session_secret="test-secret-value",
        token_encryption_key=None,
        frontend_origin="http://localhost:5173",
    )


@pytest_asyncio.fixture
async def engine(settings: Settings):  # noqa: ANN201
    # StaticPool keeps one connection so an in-memory database survives between sessions.
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:  # noqa: ANN001
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Point the app's module-level accessors at the test engine.
    session_module._engine = engine
    session_module._session_factory = factory

    yield factory

    session_module._engine = None
    session_module._session_factory = None


@pytest_asyncio.fixture
async def db(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def seeded(db: AsyncSession, settings: Settings) -> dict[str, str]:
    """A user with a linked GitHub connection, one repository and one issue."""
    user = User(id=new_id(), github_user_id=4242, github_login="diksha", display_name="Diksha")
    db.add(user)
    await db.flush()

    db.add(
        GitHubConnection(
            id=new_id(),
            user_id=user.id,
            encrypted_token=build_cipher(settings).encrypt("ghp_fake_token_value_1234567890"),
            scopes="repo",
        )
    )

    repository = Repository(
        id=new_id(),
        user_id=user.id,
        github_repo_id=99,
        owner="diksha",
        name="demo-repo",
        full_name="diksha/demo-repo",
        default_branch="main",
    )
    db.add(repository)
    await db.flush()

    issue = Issue(
        id=new_id(),
        repository_id=repository.id,
        number=7,
        title="Profile update returns 500 when email is empty",
        body="Expected a validation error, got HTTP 500.",
        state="open",
        labels=["bug"],
    )
    db.add(issue)
    await db.flush()
    await db.commit()

    return {"user_id": user.id, "repository_id": repository.id, "issue_id": issue.id}


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to the app, with settings overridden for tests."""
    from app.config.settings import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    get_settings.cache_clear()


@pytest.fixture
def auth_cookie(settings: Settings):  # noqa: ANN201
    def _make(user_id: str) -> dict[str, str]:
        token = build_session_codec(settings).issue(user_id)
        return {settings.session_cookie_name: token}

    return _make
