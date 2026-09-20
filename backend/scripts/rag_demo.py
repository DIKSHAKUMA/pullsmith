"""Indexes a directory into Neon and runs hybrid searches against it.

The end-to-end proof for Phase 2B: real embeddings, real pgvector search, real trigram
matching, real fusion. Prints what was retrieved for each query so relevance can be judged
by eye before it is measured properly.

Usage:
    python -m scripts.rag_demo <path> [--query "..."] [--limit 5]
"""

import argparse
import asyncio
import time
from pathlib import Path

from sqlalchemy import delete, select

from app.config.settings import get_settings
from app.db.base import new_id
from app.db.session import dispose_engine, init_engine, session_scope
from app.models.core import Repository, RepositorySnapshot, User
from app.models.rag import ChunkEmbedding, CodeChunk, CodeFile
from app.rag.embeddings import build_provider
from app.rag.indexer import index_snapshot
from app.rag.retrieve import hybrid_search

DEFAULT_QUERIES = [
    ("where is the run state machine defined?", []),
    ("how are secrets removed from logs?", []),
    ("how does the worker claim a job without two workers taking the same one?", []),
    ("where are github tokens encrypted?", []),
    ("ProfileService.update", ["ProfileService", "update"]),
    ("reciprocal_rank_fusion", ["reciprocal_rank_fusion"]),
]


async def ensure_fixtures(session, root: Path) -> tuple[Repository, RepositorySnapshot]:
    """Creates a local user, repository and snapshot to index into."""
    user = (
        await session.execute(select(User).where(User.github_login == "rag-demo"))
    ).scalar_one_or_none()

    if user is None:
        user = User(id=new_id(), github_user_id=999_001, github_login="rag-demo")
        session.add(user)
        await session.flush()

    repository = (
        await session.execute(
            select(Repository).where(Repository.full_name == f"local/{root.name}")
        )
    ).scalar_one_or_none()

    if repository is None:
        repository = Repository(
            id=new_id(),
            user_id=user.id,
            github_repo_id=999_001,
            owner="local",
            name=root.name,
            full_name=f"local/{root.name}",
        )
        session.add(repository)
        await session.flush()

    # A synthetic commit id so repeated runs replace the same snapshot rather than piling
    # up. A real run uses the actual commit SHA.
    commit = f"local-{root.name}"

    snapshot = (
        await session.execute(
            select(RepositorySnapshot).where(
                RepositorySnapshot.repository_id == repository.id,
                RepositorySnapshot.commit_sha == commit,
            )
        )
    ).scalar_one_or_none()

    if snapshot is not None:
        # Cascades remove chunks and embeddings too.
        await session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.snapshot_id == snapshot.id)
        )
        await session.execute(delete(CodeChunk).where(CodeChunk.snapshot_id == snapshot.id))
        await session.execute(delete(CodeFile).where(CodeFile.snapshot_id == snapshot.id))
        await session.flush()
    else:
        snapshot = RepositorySnapshot(
            id=new_id(),
            repository_id=repository.id,
            commit_sha=commit,
            index_status="pending",
        )
        session.add(snapshot)
        await session.flush()

    return repository, snapshot


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    root: Path = args.path.resolve()
    settings = get_settings()

    if settings.is_sqlite:
        raise SystemExit("this demo needs Postgres: pgvector and trigram are Postgres-only")

    init_engine(settings)

    provider = build_provider(
        provider=settings.embedding_provider,
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        requests_per_minute=settings.embedding_requests_per_minute,
    )

    print(f"provider: {provider.model} at {provider.dimensions} dimensions\n")

    async with session_scope() as session:
        _repository, snapshot = await ensure_fixtures(session, root)

        started = time.perf_counter()
        result = await index_snapshot(
            session, snapshot=snapshot, checkout_path=root, provider=provider
        )
        elapsed = time.perf_counter() - started
        await session.commit()

        print(
            f"indexed {result.files_indexed} files -> {result.chunks_created} chunks, "
            f"{result.embeddings_created} embeddings in {elapsed:.1f}s"
        )

    queries = [(query, []) for query in args.query] if args.query else DEFAULT_QUERIES

    async with session_scope() as session:
        for query, symbols in queries:
            started = time.perf_counter()

            query_vector = await provider.embed_query(query)
            results, stats = await hybrid_search(
                session,
                snapshot_id=snapshot.id,
                query=query if symbols else "",
                query_vector=query_vector,
                model=provider.model,
                symbols=symbols,
                limit=args.limit,
            )

            latency_ms = (time.perf_counter() - started) * 1000

            print(f'\n"{query}"')
            print(
                f"  {stats.semantic_candidates} semantic + {stats.lexical_candidates} "
                f"lexical -> {stats.fused_candidates} fused  ({latency_ms:.0f}ms)"
            )

            for position, item in enumerate(results, start=1):
                found_by = "+".join(sorted(set(item.strategies)))
                print(
                    f"  {position}. {item.citation():<48} {item.qualified_name():<34} "
                    f"[{found_by}] {item.score:.4f}"
                )

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
