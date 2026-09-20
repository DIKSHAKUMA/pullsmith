"""Verifies the database connection and enables pgvector.

Run this before the first migration. It confirms three things that are easy to get wrong
with a hosted Postgres and would otherwise fail confusingly later:

1. the connection string reaches the server at all (TLS, credentials, host)
2. the pgvector extension can be created
3. vector distance operators actually work

Usage:  python -m scripts.db_check
"""

import asyncio

from sqlalchemy import text

from app.config.settings import get_settings
from app.db.session import dispose_engine, init_engine


async def main() -> None:
    settings = get_settings()

    if settings.is_sqlite:
        raise SystemExit("DATABASE_URL points at SQLite; set the Neon URL first")

    engine = init_engine(settings)

    async with engine.begin() as connection:
        version = (await connection.execute(text("SELECT version()"))).scalar_one()
        print(f"connected: {str(version).split(' on ')[0]}")

        database, user = (
            await connection.execute(text("SELECT current_database(), current_user"))
        ).one()
        print(f"database : {database}  user: {user}")

        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        extension_version = (
            await connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
        ).scalar_one_or_none()
        print(f"pgvector : {extension_version}")

        # Proves the operators are usable, not just that the extension row exists.
        # CAST(... AS vector) rather than the ::vector shorthand: SQLAlchemy's text()
        # treats a leading colon as a bind parameter, so "::" collides with it.
        distance = (
            await connection.execute(
                text(
                    "SELECT ROUND(CAST(CAST(:a AS vector) <=> CAST(:b AS vector) AS numeric), 4)"
                ),
                {"a": "[1,0,0]", "b": "[0,1,0]"},
            )
        ).scalar_one()
        print(f"cosine distance of orthogonal vectors: {distance}  (expected 1.0000)")

    await dispose_engine()
    print("\nok")


if __name__ == "__main__":
    asyncio.run(main())
