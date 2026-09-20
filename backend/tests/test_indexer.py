"""Snapshot indexing, and the incremental path in particular.

The incremental path had no tests, and it was broken: unchanged files were counted and never
copied into the new snapshot. Because every chunk is scoped to one snapshot, that left a fresh
snapshot with zero chunks and retrieval that silently returned nothing. Nothing raised, nothing
logged an error, and the agent simply planned without any code in front of it.

These tests pin the behaviour the docstring always claimed: unchanged files are carried forward
with their existing vectors, and no embedding request is made for them.
"""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import new_id
from app.models.core import RepositorySnapshot
from app.models.rag import ChunkEmbedding, CodeChunk, CodeFile
from app.rag import indexer
from app.rag.embeddings import FakeEmbeddingProvider

MODULE_A = '''\
"""Order maths."""


def subtotal(items):
    return sum(item.price for item in items)


def total_with_tax(items, rate):
    return subtotal(items) * (1 + rate)
'''

MODULE_B = '''\
"""Customer records."""


class CustomerRepository:
    def find(self, customer_id):
        return self._rows.get(customer_id)

    def save(self, customer):
        self._rows[customer.id] = customer
'''


class CountingEmbeddings:
    """Wraps the deterministic fake provider and counts what it was asked to embed.

    The whole value of incremental indexing is *not* making these calls, so the count is the
    assertion that matters.
    """

    def __init__(self, *, dimensions: int = 64, model: str = "fake-embedding-v1") -> None:
        self._inner = FakeEmbeddingProvider(dimensions=dimensions, model=model)
        self.documents_embedded = 0
        self.calls = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.documents_embedded += len(texts)
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._inner.embed_query(text)


def write_project(root: Path, *, module_a: str = MODULE_A, module_b: str = MODULE_B) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "orders.py").write_text(module_a, encoding="utf-8")
    (root / "customers.py").write_text(module_b, encoding="utf-8")
    return root


async def make_snapshot(db: AsyncSession, repository_id: str, commit: str) -> RepositorySnapshot:
    snapshot = RepositorySnapshot(
        id=new_id(),
        repository_id=repository_id,
        commit_sha=commit,
        index_status="indexing",
    )
    db.add(snapshot)
    await db.flush()
    return snapshot


async def count_chunks(db: AsyncSession, snapshot_id: str) -> int:
    return (
        await db.execute(
            select(func.count()).select_from(CodeChunk).where(CodeChunk.snapshot_id == snapshot_id)
        )
    ).scalar_one()


async def count_embeddings(db: AsyncSession, snapshot_id: str) -> int:
    return (
        await db.execute(
            select(func.count())
            .select_from(ChunkEmbedding)
            .where(ChunkEmbedding.snapshot_id == snapshot_id)
        )
    ).scalar_one()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    return write_project(tmp_path / "checkout")


class TestFirstIndex:
    async def test_chunks_and_vectors_are_written(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        snapshot = await make_snapshot(db, seeded["repository_id"], "commit-one")
        provider = CountingEmbeddings()

        result = await indexer.index_snapshot(
            db, snapshot=snapshot, checkout_path=checkout, provider=provider
        )

        assert result.files_indexed == 2
        assert result.chunks_created > 0
        assert result.chunks_reused == 0
        assert result.embeddings_created == result.chunks_created

        # Every chunk has exactly one vector; a chunk without one is invisible to retrieval.
        assert await count_embeddings(db, snapshot.id) == await count_chunks(db, snapshot.id)
        assert snapshot.chunk_count == result.chunks_created
        assert snapshot.index_status == "ready"


class TestIncrementalIndex:
    async def test_unchanged_files_are_carried_forward_without_re_embedding(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        """The bug this file exists for: a new snapshot of identical content must not be empty."""
        first = await make_snapshot(db, seeded["repository_id"], "commit-one")
        provider = CountingEmbeddings()

        original = await indexer.index_snapshot(
            db, snapshot=first, checkout_path=checkout, provider=provider
        )
        embedded_after_first = provider.documents_embedded

        second = await make_snapshot(db, seeded["repository_id"], "commit-two")

        result = await indexer.index_snapshot(
            db,
            snapshot=second,
            checkout_path=checkout,
            provider=provider,
            previous_snapshot_id=first.id,
        )

        assert result.files_indexed == 0, "nothing changed, so nothing should be re-indexed"
        assert result.files_reused == 2
        assert result.chunks_reused == original.chunks_created
        assert result.embeddings_created == 0

        # The saving is real: not one extra document was sent to the embedding provider.
        assert provider.documents_embedded == embedded_after_first

        # And the new snapshot is actually usable, which is what the bug broke.
        assert await count_chunks(db, second.id) == original.chunks_created
        assert await count_embeddings(db, second.id) == original.chunks_created
        assert second.chunk_count == original.chunks_created

    async def test_only_the_changed_file_is_re_embedded(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        first = await make_snapshot(db, seeded["repository_id"], "commit-one")
        provider = CountingEmbeddings()

        await indexer.index_snapshot(
            db, snapshot=first, checkout_path=checkout, provider=provider
        )
        baseline = provider.documents_embedded

        (checkout / "orders.py").write_text(
            MODULE_A + "\n\ndef discount(items, amount):\n    return subtotal(items) - amount\n",
            encoding="utf-8",
        )

        second = await make_snapshot(db, seeded["repository_id"], "commit-two")

        result = await indexer.index_snapshot(
            db,
            snapshot=second,
            checkout_path=checkout,
            provider=provider,
            previous_snapshot_id=first.id,
        )

        assert result.files_indexed == 1
        assert result.files_reused == 1
        assert result.chunks_created > 0
        assert result.chunks_reused > 0

        # Only the edited file's chunks were embedded again.
        assert provider.documents_embedded - baseline == result.chunks_created

        # The whole snapshot is retrievable: new chunks plus carried-forward ones.
        assert await count_chunks(db, second.id) == result.total_chunks
        assert second.chunk_count == result.total_chunks

    async def test_deleted_files_do_not_reappear(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        """A snapshot must describe its own commit, so a removed file must not be carried on."""
        first = await make_snapshot(db, seeded["repository_id"], "commit-one")
        provider = CountingEmbeddings()

        await indexer.index_snapshot(
            db, snapshot=first, checkout_path=checkout, provider=provider
        )

        (checkout / "customers.py").unlink()
        second = await make_snapshot(db, seeded["repository_id"], "commit-two")

        await indexer.index_snapshot(
            db,
            snapshot=second,
            checkout_path=checkout,
            provider=provider,
            previous_snapshot_id=first.id,
        )

        paths = set(
            (
                await db.execute(
                    select(CodeFile.relative_path).where(CodeFile.snapshot_id == second.id)
                )
            )
            .scalars()
            .all()
        )

        assert paths == {"orders.py"}

    async def test_changing_the_embedding_model_forces_a_re_index(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        """Vectors from different models are not comparable, so they cannot be reused.

        Carrying them forward would silently mix two vector spaces in one snapshot and produce
        rankings that look plausible and mean nothing.
        """
        first = await make_snapshot(db, seeded["repository_id"], "commit-one")
        old_provider = CountingEmbeddings(model="fake-embedding-v1")

        original = await indexer.index_snapshot(
            db, snapshot=first, checkout_path=checkout, provider=old_provider
        )

        second = await make_snapshot(db, seeded["repository_id"], "commit-two")
        new_provider = CountingEmbeddings(model="fake-embedding-v2")

        result = await indexer.index_snapshot(
            db,
            snapshot=second,
            checkout_path=checkout,
            provider=new_provider,
            previous_snapshot_id=first.id,
        )

        assert result.files_reused == 0
        assert result.files_indexed == 2
        assert result.chunks_created == original.chunks_created
        assert new_provider.documents_embedded == original.chunks_created

        models = set(
            (
                await db.execute(
                    select(ChunkEmbedding.model).where(ChunkEmbedding.snapshot_id == second.id)
                )
            )
            .scalars()
            .all()
        )

        assert models == {"fake-embedding-v2"}

    async def test_a_previous_snapshot_with_no_index_is_re_indexed(
        self, db: AsyncSession, seeded: dict[str, str], checkout: Path
    ) -> None:
        """Pointing at a snapshot that was never indexed must not produce an empty snapshot."""
        empty = await make_snapshot(db, seeded["repository_id"], "commit-zero")
        second = await make_snapshot(db, seeded["repository_id"], "commit-one")
        provider = CountingEmbeddings()

        result = await indexer.index_snapshot(
            db,
            snapshot=second,
            checkout_path=checkout,
            provider=provider,
            previous_snapshot_id=empty.id,
        )

        assert result.files_indexed == 2
        assert result.chunks_created > 0
        assert await count_chunks(db, second.id) == result.chunks_created
