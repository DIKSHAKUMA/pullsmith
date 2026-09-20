"""Creates the schema directly from the models. Local development only.

Alembic is the real migration path and targets Neon Postgres. This script exists
because Alembic's CLI could not be run reliably on the development machine (it loads a
large import graph and the machine has 6 GB of RAM, so DLL loads intermittently fail
with E_OUTOFMEMORY). ``create_all`` needs a far smaller import graph.

Never use this against a database that has migration history: it does not record a
revision, so Alembic would then try to create tables that already exist.

Usage:  python -m scripts.init_db
"""

import asyncio

from app.config.settings import AppEnv, get_settings
from app.db.base import Base
from app.db.session import init_engine

# Import registers the models on Base.metadata.
from app.models import core, rag, review  # noqa: F401


async def main() -> None:
    settings = get_settings()

    if settings.app_env is AppEnv.production:
        raise SystemExit("init_db refuses to run in production; use alembic upgrade head")

    engine = init_engine(settings)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    await engine.dispose()

    tables = ", ".join(sorted(Base.metadata.tables))
    print(f"created {len(Base.metadata.tables)} tables: {tables}")


if __name__ == "__main__":
    asyncio.run(main())
