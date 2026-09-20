"""Phase 2 ORM models: indexed files, code chunks and their embeddings.

Everything hangs off ``repository_snapshot``, which is pinned to one commit SHA. That is
the design decision worth defending: retrieval is always scoped to the exact version of
the code the agent is editing, so it can never be handed a chunk that no longer exists.

Embeddings live in their own table rather than as a column on ``code_chunk`` for two
reasons: the embedding model can change without rewriting chunk rows, and two models can
coexist during a migration.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin
from app.db.vector_type import VectorColumn

#: Fixed at migration time, so it is a module constant rather than a setting.
#:
#: 1536 rather than the model's native 3072 because pgvector's HNSW index supports at most
#: 2000 dimensions. The model uses Matryoshka representation learning, so truncating keeps
#: most of the signal - but truncated vectors are no longer unit length and must be
#: renormalised before cosine comparison.
EMBEDDING_DIMENSIONS = 1536


class CodeFile(Base, IdMixin, TimestampMixin):
    """One indexed file at one commit."""

    __tablename__ = "code_file"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "relative_path", name="uq_file_per_snapshot"),
        Index("ix_file_snapshot", "snapshot_id"),
        Index("ix_file_language", "snapshot_id", "language"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository_snapshot.id", ondelete="CASCADE"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))

    #: SHA-256 of the file contents. Comparing this between snapshots is what makes
    #: incremental indexing possible: unchanged files keep their chunks and embeddings.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    chunks: Mapped[list["CodeChunk"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )


class CodeChunk(Base, IdMixin, TimestampMixin):
    """A syntax-bounded piece of code: a function, method, class or interface.

    The metadata columns are not decoration. ``symbol`` and ``parent_symbol`` power exact
    lookups from a stack trace, ``is_test`` lets tests be filtered in or out depending on
    the question, and the line span lets an answer cite real locations.
    """

    __tablename__ = "code_chunk"
    __table_args__ = (
        Index("ix_chunk_snapshot", "snapshot_id"),
        Index("ix_chunk_file", "file_id"),
        # Exact-symbol retrieval: "find ProfileService.update" must not scan every chunk.
        Index("ix_chunk_symbol", "snapshot_id", "symbol"),
        Index("ix_chunk_path", "snapshot_id", "relative_path"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository_snapshot.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("code_file.id", ondelete="CASCADE"), nullable=False
    )

    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))

    symbol: Mapped[str | None] = mapped_column(String(255))
    symbol_kind: Mapped[str | None] = mapped_column(String(32))
    parent_symbol: Mapped[str | None] = mapped_column(String(255))

    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    imports: Mapped[list[str] | None] = mapped_column(JSON)

    is_test: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: "syntax", "syntax-split" or "lines". Recorded so retrieval quality can be
    #: attributed to the chunking strategy instead of blamed on the embedding model.
    strategy: Mapped[str] = mapped_column(String(16), default="syntax", nullable=False)

    token_estimate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    file: Mapped[CodeFile] = relationship(back_populates="chunks")
    embedding: Mapped["ChunkEmbedding | None"] = relationship(
        back_populates="chunk", uselist=False, cascade="all, delete-orphan"
    )

    def qualified_name(self) -> str:
        if self.symbol and self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol or self.relative_path


class ChunkEmbedding(Base, IdMixin):
    """The vector for one chunk, produced by one named model."""

    __tablename__ = "chunk_embedding"
    __table_args__ = (
        UniqueConstraint("chunk_id", "model", name="uq_embedding_per_model"),
        Index("ix_embedding_snapshot", "snapshot_id"),
    )

    chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("code_chunk.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("repository_snapshot.id", ondelete="CASCADE"), nullable=False
    )

    #: Recorded so a model change is detectable and two models can coexist while
    #: re-indexing. Without this column, swapping models silently mixes incomparable
    #: vectors in one index.
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)

    vector: Mapped[list[float]] = mapped_column(
        VectorColumn(EMBEDDING_DIMENSIONS), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    chunk: Mapped[CodeChunk] = relationship(back_populates="embedding")
