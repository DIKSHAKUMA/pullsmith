"""Assembles everything a run needs, from configuration.

The orchestrator is easier to trust and far easier to test when it does not construct its own
dependencies. Everything it uses — the model, the embedding provider, the sandbox, the tool
registry, the retriever — arrives in one object built here. A test swaps in fakes by building a
different ``AgentRuntime``; the orchestrator code is identical in both cases.

The retriever is passed as a closure rather than a repository object, so the orchestrator never
holds a database session for retrieval and cannot accidentally keep one open across a long
model call.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import ToolRegistry
from app.agent.tools.editing import register_editing_tools
from app.agent.tools.filesystem import register_filesystem_tools
from app.config.settings import SandboxBackend, Settings
from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.rag.embeddings import EmbeddingProvider
from app.rag.embeddings import build_provider as build_embedding_provider
from app.rag.retrieve import RetrievedChunk, hybrid_search
from app.sandbox.base import SandboxRunner
from app.sandbox.local import LocalSubprocessSandbox

logger = logging.getLogger(__name__)

RetrieverType = Callable[[list[str], list[str]], Awaitable[list[RetrievedChunk]]]


def build_registry() -> ToolRegistry:
    """Every tool the agent can ever use.

    Write tools are registered here but marked ``mutating``, so read-only phases exclude them
    from what the model is told *and* the executor refuses them. Registration is not permission.
    """
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_editing_tools(registry)
    return registry


def build_sandbox(settings: Settings) -> SandboxRunner:
    """Selects a sandbox backend.

    Only the local runner exists today. The remote backends are designed for but not built, and
    this raises rather than silently falling back to the unisolated runner — a silent downgrade
    from "isolated" to "not isolated" is exactly the kind of thing that should be loud.
    """
    if settings.sandbox_backend is SandboxBackend.local_unsafe:
        return LocalSubprocessSandbox()

    raise NotImplementedError(
        f"SANDBOX_BACKEND={settings.sandbox_backend} is not implemented yet. "
        f"Set SANDBOX_BACKEND=local_unsafe for development, understanding that it provides "
        f"process confinement and not isolation."
    )


def build_llm(settings: Settings) -> LLMProvider:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required to run the agent")

    return GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    return build_embedding_provider(
        provider=settings.embedding_provider,
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        requests_per_minute=settings.embedding_requests_per_minute,
    )


def build_retriever(
    *,
    session_factory: Callable[[], AsyncSession],
    embeddings: EmbeddingProvider,
    snapshot_id: str,
    limit: int,
) -> RetrieverType:
    """Returns a function that turns queries into retrieved chunks.

    Each call opens its own short-lived session. Holding one open across an embedding request
    and a model call would pin a connection from Neon's small pool for the duration of a
    network round trip.
    """

    async def retrieve(queries: list[str], symbols: list[str]) -> list[RetrievedChunk]:
        if not queries:
            return []

        collected: dict[str, RetrievedChunk] = {}

        # Several queries per phase is normal: the issue analysis produces a handful, and a
        # failure produces more. Results are merged by chunk id, keeping the best score.
        for query in queries[:6]:
            vector = await embeddings.embed_query(query)

            async with session_factory() as session:
                chunks, _stats = await hybrid_search(
                    session,
                    snapshot_id=snapshot_id,
                    query=query,
                    query_vector=vector,
                    model=embeddings.model,
                    symbols=symbols,
                    limit=limit,
                )

            for chunk in chunks:
                existing = collected.get(chunk.chunk_id)

                if existing is None or chunk.score > existing.score:
                    collected[chunk.chunk_id] = chunk

        ranked = sorted(collected.values(), key=lambda item: item.score, reverse=True)

        logger.info("retrieved %s unique chunks for %s queries", len(ranked), len(queries))
        return ranked[:limit]

    return retrieve


@dataclass
class AgentRuntime:
    """Everything a run needs, injected rather than constructed in place."""

    settings: Settings
    llm: LLMProvider
    embeddings: EmbeddingProvider
    sandbox: SandboxRunner
    registry: ToolRegistry

    #: Where repositories are checked out. One directory per run.
    workspace_root: Path = field(default_factory=lambda: Path(".agent-workspaces"))

    #: How a retriever is obtained for a snapshot. Overridable because similarity search is
    #: Postgres-only: the orchestrator tests run on SQLite and substitute this, while
    #: retrieval itself is tested against a real Postgres instance.
    retriever_factory: Callable[[str], RetrieverType] | None = None

    def workspace_for(self, run_id: str) -> Path:
        return self.workspace_root / run_id


def build_runtime(settings: Settings) -> AgentRuntime:
    """The production wiring."""
    return AgentRuntime(
        settings=settings,
        llm=build_llm(settings),
        embeddings=build_embeddings(settings),
        sandbox=build_sandbox(settings),
        registry=build_registry(),
        workspace_root=Path(settings.agent_workspace_root),
    )
