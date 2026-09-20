"""phase 2 code rag tables

Revision ID: 4ac8bdcf9b19
Revises: 8c0540c85e61
Create Date: 2026-09-15 08:19:27.834348

Hand-edited after autogeneration, for three reasons Alembic cannot infer:

1. The vector column was rendered without its dimension. ``vector`` needs a fixed size,
   so it is declared explicitly as ``Vector(1536)``.
2. Vector similarity needs an **HNSW** index, which autogenerate does not know about.
   Without it every search is a sequential scan over every chunk in the repository.
3. Exact-symbol and substring search need **pg_trgm** GIN indexes. Semantic search alone
   cannot reliably find `ProfileService.update` from a stack trace, because embeddings
   match meaning rather than exact identifiers.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "4ac8bdcf9b19"
down_revision: str | None = "8c0540c85e61"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Fixed at migration time. 1536 rather than the model's native 3072 because pgvector
#: supports HNSW indexing up to 2000 dimensions. Changing this requires a new migration
#: and re-embedding every chunk.
EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    # Extensions first: the table and indexes below depend on them.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "code_file",
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("is_test", sa.Boolean(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["repository_snapshot.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("snapshot_id", "relative_path", name="uq_file_per_snapshot"),
    )
    op.create_index("ix_file_language", "code_file", ["snapshot_id", "language"], unique=False)
    op.create_index("ix_file_snapshot", "code_file", ["snapshot_id"], unique=False)
    # Content hashes are compared per snapshot during incremental indexing.
    op.create_index("ix_file_hash", "code_file", ["snapshot_id", "content_hash"], unique=False)

    op.create_table(
        "code_chunk",
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("file_id", sa.String(length=36), nullable=False),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("symbol", sa.String(length=255), nullable=True),
        sa.Column("symbol_kind", sa.String(length=32), nullable=True),
        sa.Column("parent_symbol", sa.String(length=255), nullable=True),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("imports", sa.JSON(), nullable=True),
        sa.Column("is_test", sa.Boolean(), nullable=False),
        sa.Column("strategy", sa.String(length=16), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["code_file.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["repository_snapshot.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunk_file", "code_chunk", ["file_id"], unique=False)
    op.create_index("ix_chunk_path", "code_chunk", ["snapshot_id", "relative_path"], unique=False)
    op.create_index("ix_chunk_snapshot", "code_chunk", ["snapshot_id"], unique=False)
    op.create_index("ix_chunk_symbol", "code_chunk", ["snapshot_id", "symbol"], unique=False)

    # Lexical retrieval. Trigram indexes support ILIKE and similarity() on identifiers and
    # code text, which is how a stack-trace symbol or an exact error string is found.
    # Chosen over Postgres full-text search because English stemming is wrong for code:
    # snake_case and camelCase identifiers are not words.
    op.execute(
        "CREATE INDEX ix_chunk_symbol_trgm ON code_chunk "
        "USING gin (symbol gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_chunk_content_trgm ON code_chunk "
        "USING gin (content gin_trgm_ops)"
    )

    op.create_table(
        "chunk_embedding",
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["code_chunk.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["repository_snapshot.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", "model", name="uq_embedding_per_model"),
    )
    op.create_index("ix_embedding_snapshot", "chunk_embedding", ["snapshot_id"], unique=False)

    # HNSW: a navigable small-world graph giving approximate nearest-neighbour search in
    # roughly logarithmic time instead of scanning every row.
    #
    # vector_cosine_ops matches the <=> operator, which is correct for normalised text
    # embeddings. Using the wrong operator class means the planner silently ignores the
    # index and falls back to a sequential scan.
    #
    # m=16 and ef_construction=64 are pgvector's defaults: a reasonable recall/build-time
    # trade-off. Raising them improves recall at the cost of build time and memory, which
    # is a decision to make once retrieval quality has actually been measured.
    op.execute(
        "CREATE INDEX ix_embedding_vector_hnsw ON chunk_embedding "
        "USING hnsw (vector vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embedding_vector_hnsw")
    op.drop_index("ix_embedding_snapshot", table_name="chunk_embedding")
    op.drop_table("chunk_embedding")

    op.execute("DROP INDEX IF EXISTS ix_chunk_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_chunk_symbol_trgm")
    op.drop_index("ix_chunk_symbol", table_name="code_chunk")
    op.drop_index("ix_chunk_snapshot", table_name="code_chunk")
    op.drop_index("ix_chunk_path", table_name="code_chunk")
    op.drop_index("ix_chunk_file", table_name="code_chunk")
    op.drop_table("code_chunk")

    op.drop_index("ix_file_hash", table_name="code_file")
    op.drop_index("ix_file_snapshot", table_name="code_file")
    op.drop_index("ix_file_language", table_name="code_file")
    op.drop_table("code_file")

    # Extensions are left in place: other objects may depend on them, and dropping a
    # shared extension during a rollback is a wider blast radius than this migration owns.
