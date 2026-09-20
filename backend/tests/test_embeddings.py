"""Embedding provider tests.

The Gemini provider is tested against a mock transport rather than the live API: tests must
be free, offline and deterministic. What is verified is our own behaviour - batching,
renormalisation, task types, retry policy - not Google's.
"""

import math

import httpx
import pytest

from app.rag.embeddings import (
    EmbeddingError,
    FakeEmbeddingProvider,
    GeminiEmbeddingProvider,
    build_provider,
    normalise,
)


def magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


class TestNormalise:
    def test_scales_to_unit_length(self) -> None:
        assert magnitude(normalise([3.0, 4.0])) == pytest.approx(1.0)

    def test_preserves_direction(self) -> None:
        """Only the magnitude changes; ratios between components stay the same."""
        result = normalise([3.0, 4.0])

        assert result[0] / result[1] == pytest.approx(0.75)

    def test_zero_vector_is_returned_unchanged(self) -> None:
        # Dividing by zero magnitude would raise; a zero vector has no direction to keep.
        assert normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_already_normalised_vector_is_stable(self) -> None:
        once = normalise([1.0, 2.0, 3.0])
        twice = normalise(once)

        assert once == pytest.approx(twice)


class TestFakeProvider:
    async def test_is_deterministic(self) -> None:
        """Reproducibility is the whole point: a real model cannot give this."""
        provider = FakeEmbeddingProvider(dimensions=64)

        first = await provider.embed_query("def validate_email(email): ...")
        second = await provider.embed_query("def validate_email(email): ...")

        assert first == second

    async def test_returns_requested_dimensions(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=128)

        vectors = await provider.embed_documents(["alpha", "beta"])

        assert [len(vector) for vector in vectors] == [128, 128]

    async def test_vectors_are_unit_length(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=64)

        vector = await provider.embed_query("some code")

        assert magnitude(vector) == pytest.approx(1.0)

    async def test_similar_text_scores_higher_than_unrelated_text(self) -> None:
        """Enough signal to exercise ranking logic, without being a semantic model."""
        provider = FakeEmbeddingProvider(dimensions=512)

        query = await provider.embed_query("validate email address")
        related = await provider.embed_query("validate email addresses properly")
        unrelated = await provider.embed_query("zzz qqq xxx")

        def cosine(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True))

        assert cosine(query, related) > cosine(query, unrelated)

    async def test_empty_input_returns_empty_list(self) -> None:
        assert await FakeEmbeddingProvider().embed_documents([]) == []

    async def test_empty_string_does_not_produce_a_zero_vector(self) -> None:
        """A zero vector makes cosine similarity undefined."""
        vector = await FakeEmbeddingProvider(dimensions=32).embed_query("")

        assert magnitude(vector) == pytest.approx(1.0)


def gemini_transport(
    *,
    dimensions: int = 8,
    magnitude_scale: float = 0.7,
    capture: list[httpx.Request] | None = None,
    statuses: list[int] | None = None,
) -> httpx.MockTransport:
    """Mimics the API, including its non-unit-length truncated vectors."""
    remaining_statuses = list(statuses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)

        if remaining_statuses:
            status = remaining_statuses.pop(0)
            if status >= 400:
                return httpx.Response(status, json={"error": {"message": "transient"}})

        # Deliberately not unit length, matching the real API's truncated output.
        values = [magnitude_scale / math.sqrt(dimensions)] * dimensions

        if "batchEmbedContents" in str(request.url):
            count = len(request.read().decode().split('"parts"')) - 1
            return httpx.Response(
                200, json={"embeddings": [{"values": values} for _ in range(count)]}
            )

        return httpx.Response(200, json={"embedding": {"values": values}})

    return httpx.MockTransport(handler)


async def _no_sleep(_seconds: float) -> None:
    """Keeps retry tests instant instead of actually waiting out the backoff."""
    return None


class TestGeminiProvider:
    async def test_requires_an_api_key(self) -> None:
        with pytest.raises(EmbeddingError, match="API key"):
            GeminiEmbeddingProvider(api_key="")

    async def test_renormalises_truncated_vectors(self) -> None:
        """The bug this guards against is silent.

        Truncated Gemini vectors are not unit length (measured 0.6962 at 1536 dimensions).
        Cosine similarity would still return numbers, just subtly wrong ones, so nothing
        would fail loudly.
        """
        provider = GeminiEmbeddingProvider(
            api_key="test-key",
            dimensions=8,
            transport=gemini_transport(dimensions=8, magnitude_scale=0.6962),
        )

        vector = await provider.embed_query("where is email validation?")

        assert magnitude(vector) == pytest.approx(1.0, abs=1e-6)

    async def test_documents_are_batched(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiEmbeddingProvider(
            api_key="test-key",
            dimensions=8,
            batch_size=4,
            transport=gemini_transport(dimensions=8, capture=captured),
        )

        vectors = await provider.embed_documents([f"text {index}" for index in range(10)])

        assert len(vectors) == 10
        # 10 texts at batch size 4 is three requests, not ten.
        assert len(captured) == 3
        assert all("batchEmbedContents" in str(request.url) for request in captured)

    async def test_query_and_document_use_different_task_types(self) -> None:
        """Queries and passages are encoded differently; using one type for both hurts."""
        captured: list[httpx.Request] = []
        provider = GeminiEmbeddingProvider(
            api_key="test-key",
            dimensions=8,
            transport=gemini_transport(dimensions=8, capture=captured),
        )

        await provider.embed_documents(["some code"])
        await provider.embed_query("a question")

        bodies = [request.read().decode() for request in captured]

        assert "RETRIEVAL_DOCUMENT" in bodies[0]
        assert "RETRIEVAL_QUERY" in bodies[1]

    async def test_requested_dimensions_are_sent(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiEmbeddingProvider(
            api_key="k", dimensions=8, transport=gemini_transport(dimensions=8, capture=captured)
        )

        await provider.embed_query("q")

        assert "outputDimensionality" in captured[0].read().decode()

    async def test_transient_failure_is_retried(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr("app.rag.embeddings.asyncio.sleep", _no_sleep)

        provider = GeminiEmbeddingProvider(
            api_key="k",
            dimensions=8,
            max_attempts=3,
            transport=gemini_transport(dimensions=8, statuses=[503, 200]),
        )

        vector = await provider.embed_query("q")

        assert magnitude(vector) == pytest.approx(1.0, abs=1e-6)

    async def test_permanent_failure_is_not_retried(self) -> None:
        """A 400 is a real answer. Retrying it wastes quota and hides the error."""
        captured: list[httpx.Request] = []
        provider = GeminiEmbeddingProvider(
            api_key="k",
            dimensions=8,
            max_attempts=3,
            transport=gemini_transport(dimensions=8, capture=captured, statuses=[400]),
        )

        with pytest.raises(EmbeddingError, match="400"):
            await provider.embed_query("q")

        assert len(captured) == 1

    async def test_mismatched_batch_response_is_rejected(self) -> None:
        """Silently returning fewer vectors than chunks would misalign every embedding."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}]})

        provider = GeminiEmbeddingProvider(
            api_key="k", dimensions=2, transport=httpx.MockTransport(handler)
        )

        with pytest.raises(EmbeddingError, match="expected 3 embeddings"):
            await provider.embed_documents(["a", "b", "c"])

    async def test_empty_document_list_makes_no_request(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiEmbeddingProvider(
            api_key="k", transport=gemini_transport(capture=captured)
        )

        assert await provider.embed_documents([]) == []
        assert captured == []


class TestBuildProvider:
    def test_builds_fake(self) -> None:
        provider = build_provider(
            provider="fake", api_key=None, model="x", dimensions=64, batch_size=8
        )

        assert provider.dimensions == 64

    def test_builds_gemini(self) -> None:
        provider = build_provider(
            provider="gemini",
            api_key="key",
            model="gemini-embedding-001",
            dimensions=1536,
            batch_size=32,
        )

        assert provider.model == "gemini-embedding-001"

    def test_gemini_without_key_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
            build_provider(
                provider="gemini", api_key=None, model="m", dimensions=1536, batch_size=32
            )

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="unknown embedding provider"):
            build_provider(
                provider="cohere", api_key="k", model="m", dimensions=1536, batch_size=32
            )
