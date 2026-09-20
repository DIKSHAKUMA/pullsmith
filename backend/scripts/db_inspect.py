"""Prints the live schema: tables, row counts, indexes and migration revision.

A read-only check used to confirm a migration produced what was expected.

Usage:  python -m scripts.db_inspect
"""

import asyncio

from sqlalchemy import text

from app.config.settings import get_settings
from app.db.session import dispose_engine, init_engine

TABLES_QUERY = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name
"""

INDEX_QUERY = """
SELECT tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname
"""


async def main() -> None:
    settings = get_settings()
    engine = init_engine(settings)

    async with engine.connect() as connection:
        revision = (
            await connection.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one_or_none()
        print(f"alembic revision: {revision}\n")

        tables = [row[0] for row in await connection.execute(text(TABLES_QUERY))]

        print(f"tables ({len(tables)}):")
        for table in tables:
            # A table name cannot be a bind parameter in SQL, so it must be interpolated.
            # This is safe here because the value came from information_schema rather than
            # from a user, and it is validated against an identifier pattern first.
            if not table.replace("_", "").isalnum():
                print(f"  {table:<26} skipped (unexpected identifier)")
                continue

            count = (
                await connection.execute(text(f'SELECT count(*) FROM "{table}"'))  # noqa: S608
            ).scalar_one()
            print(f"  {table:<26} rows={count}")

        indexes = list(await connection.execute(text(INDEX_QUERY)))
        print(f"\nindexes ({len(indexes)}):")
        for table_name, index_name in indexes:
            print(f"  {table_name:<26} {index_name}")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
