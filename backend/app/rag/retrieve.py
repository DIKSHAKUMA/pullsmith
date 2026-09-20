"""Hybrid code retrieval.

Semantic search alone is not enough for code, and neither is keyword search.

* **Semantic** (pgvector, cosine): finds code by meaning. "where is email validation?"
  matches ``def check_address(...)`` even with no shared words. It is weak on exact
  identifiers, because an embedding of ``ProfileService.update`` is close to every other
  update method in the repository.
* **Lexical** (trigram): finds exact identifiers, error strings and paths. A stack trace
  names symbols precisely, and precision is exactly what embeddings blur.

Results are combined with **Reciprocal Rank Fusion**, which merges by *rank* rather than
score. That matters because a cosine distance and a trigram similarity are not on a
comparable scale, and normalising them against each other would be arbitrary.

Every query is scoped to one snapshot, so retrieval can only ever return code from the
exact commit the agent is working on.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import Float, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag import ChunkEmbedding, CodeChunk

logger = logging.getLogger(__name__)

#: RFF damping constant. 60 is the value from the original paper and is standard: it stops
#: a single first-place hit from dominating, so agreement between strategies wins over one
#: strategy being confident.
RRF_K = 60

#: Candidates fetched per strategy before fusion. Wider than the final result count so
#: fusion has something to work with; a chunk ranked 8th by vectors and 9th by keywords
#: should be able to beat one ranked 1st by only one of them.
CANDIDATES_PER_STRATEGY = 30


@dataclass
class RetrievedChunk:
    chunk_id: str
    relative_path: str
    symbol: str | None
    parent_symbol: str | None
    symbol_kind: str | None
    start_line: int
    end_line: int
    content: str
    language: str | None
    is_test: bool
    score: float
    strategies: list[str] = field(default_factory=list)
    semantic_rank: int | None = None
    lexical_rank: int | None = None

    def qualified_name(self) -> str:
        if self.symbol and self.parent_symbol:
            return f"{self.parent_symbol}.{self.symbol}"
        return self.symbol or self.relative_path

    def citation(self) -> str:
        return f"{self.relative_path}:{self.start_line}-{self.end_line}"


@dataclass
class RetrievalStats:
    """Recorded per query so retrieval can be debugged and evaluated later."""

    semantic_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    returned: int = 0


def _row_to_chunk(chunk: CodeChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        relative_path=chunk.relative_path,
        symbol=chunk.symbol,
        parent_symbol=chunk.parent_symbol,
        symbol_kind=chunk.symbol_kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=chunk.content,
        language=chunk.language,
        is_test=chunk.is_test,
        score=score,
    )


async def semantic_search(
    session: AsyncSession,
    *,
    snapshot_id: str,
    query_vector: list[float],
    model: str,
    limit: int = CANDIDATES_PER_STRATEGY,
    include_tests: bool = True,
) -> list[tuple[CodeChunk, float]]:
    """Nearest neighbours by cosine distance.

    Filtering by ``model`` matters: vectors from different embedding models are not
    comparable, so mixing them would silently produce nonsense rankings during a model
    migration.
    """
    distance = ChunkEmbedding.vector.cosine_distance(query_vector)

    statement = (
        select(CodeChunk, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == CodeChunk.id)
        .where(
            CodeChunk.snapshot_id == snapshot_id,
            ChunkEmbedding.model == model,
        )
        .order_by(distance)
        .limit(limit)
    )

    if not include_tests:
        statement = statement.where(CodeChunk.is_test.is_(False))

    rows = (await session.execute(statement)).all()

    # Cosine distance runs 0 (identical) to 2 (opposite); convert to a similarity so
    # higher always means better everywhere else in the pipeline.
    return [(chunk, 1.0 - float(distance)) for chunk, distance in rows]


async def lexical_search(
    session: AsyncSession,
    *,
    snapshot_id: str,
    query: str,
    symbols: list[str] | None = None,
    limit: int = CANDIDATES_PER_STRATEGY,
    include_tests: bool = True,
) -> list[tuple[CodeChunk, float]]:
    """Exact and near-exact matching on symbols, paths and content.

    Symbol names extracted from a stack trace are matched first and weighted highest,
    because an exact identifier match is a much stronger signal than fuzzy text overlap.
    """
    terms = [term for term in (symbols or []) if term]

    conditions = []
    ranking_terms = []

    for term in terms:
        # Exact symbol match, then qualified-name and path containment.
        conditions.append(func.lower(CodeChunk.symbol) == term.lower())
        conditions.append(CodeChunk.content.ilike(f"%{term}%"))
        conditions.append(CodeChunk.relative_path.ilike(f"%{term}%"))
        ranking_terms.append(term)

    if query.strip():
        conditions.append(CodeChunk.content.ilike(f"%{query.strip()}%"))

    if not conditions:
        return []

    # Trigram similarity on the symbol gives a graded score rather than a boolean, so
    # near-misses (update vs updateProfile) still rank.
    similarity = func.greatest(
        *[
            func.similarity(func.coalesce(CodeChunk.symbol, ""), term)
            for term in ranking_terms
        ]
    ) if ranking_terms else cast(0.0, Float)

    statement = (
        select(CodeChunk, similarity.label("similarity"))
        .where(CodeChunk.snapshot_id == snapshot_id, or_(*conditions))
        .order_by(similarity.desc(), CodeChunk.start_line)
        .limit(limit)
    )

    if not include_tests:
        statement = statement.where(CodeChunk.is_test.is_(False))

    rows = (await session.execute(statement)).all()
    return [(chunk, float(score or 0.0)) for chunk, score in rows]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[CodeChunk, float]]],
    *,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Merges ranked lists by position rather than by score.

    ``score = sum over lists of weight / (K + rank)``

    Using rank sidesteps the fact that cosine distance and trigram similarity are on
    different, non-comparable scales. A chunk found by *both* strategies outranks one that
    only one strategy loved, which is the behaviour we want: agreement is evidence.
    """
    weights = weights or {}
    merged: dict[str, RetrievedChunk] = {}

    for strategy, results in ranked_lists.items():
        weight = weights.get(strategy, 1.0)

        for position, (chunk, _score) in enumerate(results, start=1):
            contribution = weight / (RRF_K + position)

            existing = merged.get(chunk.id)

            if existing is None:
                existing = _row_to_chunk(chunk, 0.0)
                merged[chunk.id] = existing

            existing.score += contribution
            existing.strategies.append(strategy)

            if strategy == "semantic":
                existing.semantic_rank = position
            elif strategy == "lexical":
                existing.lexical_rank = position

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


async def hybrid_search(
    session: AsyncSession,
    *,
    snapshot_id: str,
    query: str,
    query_vector: list[float],
    model: str,
    symbols: list[str] | None = None,
    limit: int = 10,
    include_tests: bool = True,
) -> tuple[list[RetrievedChunk], RetrievalStats]:
    """Runs both strategies and fuses the results."""
    semantic = await semantic_search(
        session,
        snapshot_id=snapshot_id,
        query_vector=query_vector,
        model=model,
        include_tests=include_tests,
    )

    lexical = await lexical_search(
        session,
        snapshot_id=snapshot_id,
        query=query,
        symbols=symbols,
        include_tests=include_tests,
    )

    fused = reciprocal_rank_fusion(
        {"semantic": semantic, "lexical": lexical},
        # Lexical hits are weighted slightly higher because an exact identifier match is a
        # stronger signal than semantic similarity when the query names a symbol.
        weights={"semantic": 1.0, "lexical": 1.2},
    )

    stats = RetrievalStats(
        semantic_candidates=len(semantic),
        lexical_candidates=len(lexical),
        fused_candidates=len(fused),
        returned=min(limit, len(fused)),
    )

    logger.info(
        "retrieval: %s semantic + %s lexical -> %s fused, returning %s",
        stats.semantic_candidates,
        stats.lexical_candidates,
        stats.fused_candidates,
        stats.returned,
    )

    return fused[:limit], stats
