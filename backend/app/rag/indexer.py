"""Persists a repository snapshot: scan, chunk, embed, store.

Ordering matters here. Chunks are written first and embeddings second, so a crash between
the two leaves chunks with no embedding rather than embeddings pointing at nothing. On the
next run the missing embeddings are simply filled in, which makes the whole operation
resumable rather than all-or-nothing.

Incremental indexing compares per-file content hashes against the previous snapshot.
Unchanged files are copied forward with their existing embeddings, so re-indexing after a
one-line change costs one file rather than the whole repository.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id, utc_now
from app.models.core import RepositorySnapshot
from app.models.rag import ChunkEmbedding, CodeChunk, CodeFile
from app.rag import repo_map, workspace
from app.rag.chunk import CodeChunk as ParsedChunk
from app.rag.chunk import chunk_file
from app.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

#: Rough characters-per-token ratio for code. Used only for budgeting context later, so an
#: approximation is fine and avoids shipping a tokeniser for every model.
CHARS_PER_TOKEN = 4


#: Batch size for `IN (...)` lookups. SQLite caps bound parameters at 999, and a large
#: repository easily has more unchanged files than that.
LOOKUP_BATCH = 400


@dataclass
class IndexResult:
    snapshot_id: str
    files_indexed: int
    files_reused: int
    chunks_created: int
    chunks_reused: int
    embeddings_created: int
    embeddings_reused: int
    embedding_model: str

    @property
    def total_chunks(self) -> int:
        """Everything retrievable in this snapshot, freshly embedded or carried forward."""
        return self.chunks_created + self.chunks_reused


def _embedding_text(chunk: ParsedChunk) -> str:
    """Builds the text that actually gets embedded.

    The raw code alone is a weaker signal than code plus its location and symbol name. A
    question like "where is profile update validation?" matches the path and symbol as much
    as the body, so those are included as a short header.
    """
    header_parts = [chunk.relative_path]

    if chunk.parent_symbol and chunk.symbol:
        header_parts.append(f"{chunk.parent_symbol}.{chunk.symbol}")
    elif chunk.symbol:
        header_parts.append(chunk.symbol)

    if chunk.symbol_kind:
        header_parts.append(chunk.symbol_kind)

    if chunk.is_test:
        header_parts.append("test")

    return " | ".join(header_parts) + "\n" + chunk.content


@dataclass
class _Reused:
    files: int = 0
    chunks: int = 0
    embeddings: int = 0

    #: Unchanged files that could not be carried forward and must be re-indexed.
    stale_paths: list[str] = field(default_factory=list)


def _batched(items: list[str]) -> list[list[str]]:
    return [items[start : start + LOOKUP_BATCH] for start in range(0, len(items), LOOKUP_BATCH)]


async def _carry_forward(
    session: AsyncSession,
    *,
    snapshot: RepositorySnapshot,
    previous_snapshot_id: str | None,
    unchanged_paths: list[str],
    model: str,
) -> _Reused:
    """Copies unchanged files, their chunks and their vectors into the new snapshot.

    This is what makes incremental indexing worth having. Everything is scoped to one
    snapshot so retrieval can never return code from a commit the agent is not working on —
    which means a new snapshot starts empty, and unchanged files have to be *copied* rather
    than merely counted. Skipping this step leaves a snapshot with no chunks and retrieval
    that silently returns nothing.

    Only the chunks are copied. The embedding vector is reused as-is, which is the actual
    saving: re-embedding is the slow, paid part, re-inserting a row is not.

    A file is only reusable if every one of its chunks has a vector from the **current**
    model. Vectors from different models are not comparable, so after a model change the
    file is reported as stale and re-indexed instead.
    """
    result = _Reused()

    if not previous_snapshot_id or not unchanged_paths:
        result.stale_paths = list(unchanged_paths)
        return result

    for batch in _batched(unchanged_paths):
        files = (
            (
                await session.execute(
                    select(CodeFile).where(
                        CodeFile.snapshot_id == previous_snapshot_id,
                        CodeFile.relative_path.in_(batch),
                    )
                )
            )
            .scalars()
            .all()
        )

        rows = (
            await session.execute(
                select(CodeChunk, ChunkEmbedding)
                .outerjoin(ChunkEmbedding, ChunkEmbedding.chunk_id == CodeChunk.id)
                .where(
                    CodeChunk.snapshot_id == previous_snapshot_id,
                    CodeChunk.relative_path.in_(batch),
                )
            )
        ).all()

        chunks_by_path: dict[str, list[tuple[CodeChunk, ChunkEmbedding | None]]] = {}

        for chunk, embedding in rows:
            usable = embedding if embedding is not None and embedding.model == model else None
            chunks_by_path.setdefault(chunk.relative_path, []).append((chunk, usable))

        for old_file in files:
            pairs = chunks_by_path.get(old_file.relative_path, [])

            # Built as a separate list of non-None pairs rather than checked with `any(...)`,
            # so the reuse loop below is provably working with a vector for every chunk.
            usable = [
                (chunk, embedding) for chunk, embedding in pairs if embedding is not None
            ]

            if not pairs or len(usable) != len(pairs):
                result.stale_paths.append(old_file.relative_path)
                continue

            new_file = CodeFile(
                id=new_id(),
                snapshot_id=snapshot.id,
                relative_path=old_file.relative_path,
                language=old_file.language,
                content_hash=old_file.content_hash,
                size_bytes=old_file.size_bytes,
                line_count=old_file.line_count,
                is_test=old_file.is_test,
                chunk_count=old_file.chunk_count,
            )
            session.add(new_file)
            result.files += 1

            for old_chunk, old_embedding in usable:
                new_chunk = CodeChunk(
                    id=new_id(),
                    snapshot_id=snapshot.id,
                    file_id=new_file.id,
                    relative_path=old_chunk.relative_path,
                    language=old_chunk.language,
                    symbol=old_chunk.symbol,
                    symbol_kind=old_chunk.symbol_kind,
                    parent_symbol=old_chunk.parent_symbol,
                    start_line=old_chunk.start_line,
                    end_line=old_chunk.end_line,
                    content=old_chunk.content,
                    imports=old_chunk.imports,
                    is_test=old_chunk.is_test,
                    strategy=old_chunk.strategy,
                    token_estimate=old_chunk.token_estimate,
                )
                session.add(new_chunk)

                session.add(
                    ChunkEmbedding(
                        id=new_id(),
                        chunk_id=new_chunk.id,
                        snapshot_id=snapshot.id,
                        model=old_embedding.model,
                        dimensions=old_embedding.dimensions,
                        # The point of the whole exercise: no embedding request is made.
                        vector=old_embedding.vector,
                        created_at=utc_now(),
                    )
                )

                result.chunks += 1
                result.embeddings += 1

        # Unchanged paths with no row at all in the previous snapshot (it was never indexed
        # successfully) must also be re-indexed.
        indexed_paths = {item.relative_path for item in files}
        result.stale_paths.extend(path for path in batch if path not in indexed_paths)

    await session.flush()

    logger.info(
        "carried forward %s file(s), %s chunk(s), %s vector(s)",
        result.files,
        result.chunks,
        result.embeddings,
    )
    return result


async def index_snapshot(
    session: AsyncSession,
    *,
    snapshot: RepositorySnapshot,
    checkout_path: Path,
    provider: EmbeddingProvider,
    previous_snapshot_id: str | None = None,
) -> IndexResult:
    """Indexes a checkout into the given snapshot."""
    scanned = workspace.scan(checkout_path)
    mapping = repo_map.build(checkout_path, scanned)

    previous_hashes: dict[str, str] = {}

    if previous_snapshot_id:
        rows = await session.execute(
            select(CodeFile.relative_path, CodeFile.content_hash).where(
                CodeFile.snapshot_id == previous_snapshot_id
            )
        )
        previous_hashes = {path: content_hash for path, content_hash in rows.all()}

    modified, unchanged, deleted = workspace.changed_files(previous_hashes, scanned)

    # Carry unchanged files forward before anything else, because a file that cannot be
    # carried forward has to be re-indexed and therefore belongs in `modified`.
    reused = await _carry_forward(
        session,
        snapshot=snapshot,
        previous_snapshot_id=previous_snapshot_id,
        unchanged_paths=unchanged,
        model=provider.model,
    )

    if reused.stale_paths:
        # Content is identical but there is no usable embedding, which happens after an
        # embedding-model change. Re-indexing is the only correct answer.
        logger.info(
            "re-indexing %s unchanged file(s) with no usable vector", len(reused.stale_paths)
        )
        by_path = {item.relative_path: item for item in scanned}
        modified = modified + [by_path[path] for path in reused.stale_paths if path in by_path]

    logger.info(
        "indexing %s: %s to index, %s carried forward, %s deleted",
        snapshot.commit_sha[:8],
        len(modified),
        reused.files,
        len(deleted),
    )

    chunk_rows: list[CodeChunk] = []
    texts: list[str] = []

    for scanned_file in modified:
        data = workspace.read_bytes(scanned_file.absolute_path)

        if data is None:
            continue

        content = data.decode("utf-8", errors="replace")

        parsed = chunk_file(
            content=content,
            relative_path=scanned_file.relative_path,
            language=scanned_file.language,
            is_test=scanned_file.is_test,
        )

        file_row = CodeFile(
            id=new_id(),
            snapshot_id=snapshot.id,
            relative_path=scanned_file.relative_path,
            language=scanned_file.language,
            content_hash=scanned_file.content_hash,
            size_bytes=scanned_file.size_bytes,
            line_count=scanned_file.line_count,
            is_test=scanned_file.is_test,
            chunk_count=len(parsed),
        )
        session.add(file_row)

        for item in parsed:
            chunk_rows.append(
                CodeChunk(
                    id=new_id(),
                    snapshot_id=snapshot.id,
                    file_id=file_row.id,
                    relative_path=item.relative_path,
                    language=item.language,
                    symbol=item.symbol,
                    symbol_kind=item.symbol_kind,
                    parent_symbol=item.parent_symbol,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    content=item.content,
                    imports=item.imports,
                    is_test=item.is_test,
                    strategy=item.strategy,
                    token_estimate=len(item.content) // CHARS_PER_TOKEN,
                )
            )
            texts.append(_embedding_text(item))

    session.add_all(chunk_rows)
    await session.flush()

    # Embeddings second: a crash here leaves chunks without vectors, which the next run
    # can fill in. The reverse order would leave orphaned vectors.
    vectors = await provider.embed_documents(texts)

    now = utc_now()
    session.add_all(
        [
            ChunkEmbedding(
                id=new_id(),
                chunk_id=chunk.id,
                snapshot_id=snapshot.id,
                model=provider.model,
                dimensions=provider.dimensions,
                vector=vector,
                created_at=now,
            )
            for chunk, vector in zip(chunk_rows, vectors, strict=True)
        ]
    )

    snapshot.index_status = "ready"
    snapshot.indexed_at = now
    snapshot.file_count = len(scanned)

    # Must count carried-forward chunks too. This value is what the orchestrator checks to
    # decide a commit is already indexed, so undercounting it would make a usable snapshot
    # look empty and trigger a pointless re-index.
    snapshot.chunk_count = len(chunk_rows) + reused.chunks

    snapshot.embedding_model = provider.model
    snapshot.repository_map = mapping.to_dict()

    await session.flush()

    return IndexResult(
        snapshot_id=snapshot.id,
        files_indexed=len(modified),
        files_reused=reused.files,
        chunks_created=len(chunk_rows),
        chunks_reused=reused.chunks,
        embeddings_created=len(vectors),
        embeddings_reused=reused.embeddings,
        embedding_model=provider.model,
    )
