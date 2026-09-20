"""A vector column that also works on SQLite.

pgvector's ``Vector`` type only exists on Postgres. If the models used it directly, the
whole test suite would need a live Postgres connection, and a slow remote database on a
2-core laptop is the fastest way to stop running tests.

This ``TypeDecorator`` emits a real ``vector(n)`` column on Postgres and falls back to
JSON elsewhere. Storage is portable; *similarity search* is not, because the ``<=>``
operator and the HNSW index are Postgres-only. Retrieval tests therefore run against
Postgres and are marked, while everything around them stays fast on SQLite.
"""

from typing import Any

from sqlalchemy import JSON, Float
from sqlalchemy.types import TypeDecorator


class VectorColumn(TypeDecorator):
    """Stores a list of floats as ``vector(n)`` on Postgres, JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    class comparator_factory(TypeDecorator.Comparator):  # noqa: N801 - SQLAlchemy's name
        """Exposes pgvector's distance operators through the wrapper type.

        A ``TypeDecorator`` does not inherit the comparator of the type it delegates to, so
        without this the ORM attribute has no ``cosine_distance`` and every similarity
        query fails at attribute access. Declaring the operators here keeps query code
        reading as ``ChunkEmbedding.vector.cosine_distance(...)`` rather than raw SQL.

        These emit Postgres-only operators. On SQLite the column still stores and loads,
        but similarity search is unavailable - which is why retrieval tests run against
        Postgres.
        """

        def cosine_distance(self, other: object) -> Any:
            """``<=>`` - correct for normalised text embeddings."""
            return self.op("<=>", return_type=Float)(other)

        def l2_distance(self, other: object) -> Any:
            """``<->`` - straight-line distance; sensitive to magnitude."""
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other: object) -> Any:
            """``<#>`` - negative inner product."""
            return self.op("<#>", return_type=Float)(other)

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dimensions))

        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None

        if dialect.name == "postgresql":
            # pgvector's own type handles the conversion.
            return value

        return list(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None

        # Postgres returns a numpy array; normalise to a plain list so callers do not
        # need to care which backend they are on.
        return [float(item) for item in value]
