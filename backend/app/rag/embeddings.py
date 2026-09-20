"""Embedding providers.

The interface exists so the rest of the system never imports a vendor SDK. Three
implementations matter:

* ``GeminiEmbeddingProvider`` - the real one.
* ``FakeEmbeddingProvider``   - deterministic, offline, free. Retrieval logic and the
  whole indexing pipeline can be tested without spending anything or needing a network.
* anything added later          - swapping providers must not touch retrieval code.

Two behaviours are non-obvious and both are load-bearing:

**Renormalisation.** ``gemini-embedding-001`` returns unit-length vectors at its native
3072 dimensions, but truncated output is *not* unit length (measured: 0.6962 at 1536).
Cosine similarity assumes unit vectors and will silently return skewed scores otherwise -
no error, just quietly worse retrieval. Every provider therefore normalises before
returning.

**Task type.** The API distinguishes indexing a document from embedding a search query.
Using the wrong one degrades ranking, so the two calls are separate methods rather than a
flag a caller can forget.
"""

import asyncio
import hashlib
import logging
import math
import struct
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

#: Retry only transient conditions. A 400 means the request is wrong and a 403 means the
#: key is wrong; retrying either wastes quota and delays a real error reaching the user.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    """What the indexer and the retriever are allowed to depend on."""

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def normalise(vector: list[float]) -> list[float]:
    """Scales a vector to unit length.

    Required after dimension truncation. Cosine similarity divides by vector magnitude, so
    non-unit vectors do not break it outright, but mixing magnitudes across a corpus
    distorts comparisons in ways that are invisible until retrieval quality is measured.
    """
    magnitude = math.sqrt(sum(value * value for value in vector))

    if magnitude == 0.0:
        return vector

    return [value / magnitude for value in vector]


class GeminiEmbeddingProvider:
    """Google AI Studio embeddings, batched, with bounded retries."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = 1536,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_attempts: int = 4,
        requests_per_minute: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError("Gemini API key is required")

        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._timeout = timeout
        self._max_attempts = max_attempts

        #: Client-side pacing. Retrying after a 429 is reactive and wasteful: the request
        #: was still counted against quota. Spacing requests out avoids provoking the limit
        #: in the first place. 0 disables pacing.
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_request_at = 0.0

        #: Injection seam for tests. Passing a mock transport is cleaner than patching
        #: httpx globally, which recurses if the patch itself constructs a client.
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    async def _pace(self) -> None:
        """Waits, if needed, to stay under the configured requests-per-minute."""
        if self._min_interval <= 0.0:
            return

        elapsed = asyncio.get_running_loop().time() - self._last_request_at
        wait = self._min_interval - elapsed

        if wait > 0:
            await asyncio.sleep(wait)

        self._last_request_at = asyncio.get_running_loop().time()

    async def _post(self, client: httpx.AsyncClient, path: str, body: dict) -> dict:
        delay = 1.0

        for attempt in range(1, self._max_attempts + 1):
            await self._pace()

            try:
                response = await client.post(
                    f"{GEMINI_BASE}{path}", headers=self._headers(), json=body
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == self._max_attempts:
                    raise EmbeddingError(f"transport failure: {type(exc).__name__}") from exc

                await asyncio.sleep(delay)
                delay *= 2
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self._max_attempts:
                # Honour Retry-After when the server states how long to wait.
                wait = float(response.headers.get("Retry-After", delay))
                logger.warning(
                    "embedding request returned %s, retrying in %.1fs (attempt %s/%s)",
                    response.status_code,
                    wait,
                    attempt,
                    self._max_attempts,
                )
                await asyncio.sleep(wait)
                delay *= 2
                continue

            if response.status_code >= 400:
                # Truncated: an error body can echo request content.
                raise EmbeddingError(
                    f"embedding request failed with {response.status_code}: "
                    f"{response.text[:200]}"
                )

            return response.json()

        raise EmbeddingError("retry loop exhausted")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds many texts, batching requests.

        Measured on this project: 16 texts cost 8.90s sequentially and 1.58s batched, a
        5.6x saving for identical token spend, because per-request latency dominates.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []

        async with self._client() as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]

                payload = await self._post(
                    client,
                    f"/models/{self._model}:batchEmbedContents",
                    {
                        "requests": [
                            {
                                "model": f"models/{self._model}",
                                "content": {"parts": [{"text": text}]},
                                "outputDimensionality": self._dimensions,
                                "taskType": "RETRIEVAL_DOCUMENT",
                            }
                            for text in batch
                        ]
                    },
                )

                embeddings = payload.get("embeddings", [])

                if len(embeddings) != len(batch):
                    raise EmbeddingError(
                        f"expected {len(batch)} embeddings, received {len(embeddings)}"
                    )

                vectors.extend(normalise(item["values"]) for item in embeddings)

        logger.info("embedded %s texts with %s", len(vectors), self._model)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a search query.

        ``RETRIEVAL_QUERY`` rather than ``RETRIEVAL_DOCUMENT``: the model encodes a
        question differently from the passage that answers it, and using the document task
        type for queries measurably degrades ranking.
        """
        async with self._client() as client:
            payload = await self._post(
                client,
                f"/models/{self._model}:embedContent",
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self._dimensions,
                    "taskType": "RETRIEVAL_QUERY",
                },
            )

        values = payload.get("embedding", {}).get("values")

        if not values:
            raise EmbeddingError("embedding response contained no values")

        return normalise(values)


class FakeEmbeddingProvider:
    """Deterministic offline embeddings for tests.

    Not a semantic model. It hashes character trigrams into a fixed number of buckets, so
    texts sharing substrings land near each other. That is enough to exercise indexing,
    storage, ranking and fusion logic without a network call or a bill, and it makes tests
    reproducible in a way a real model cannot be.
    """

    def __init__(self, *, dimensions: int = 1536, model: str = "fake-embedding-v1") -> None:
        self._dimensions = dimensions
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * self._dimensions
        lowered = text.lower()

        for index in range(max(len(lowered) - 2, 1)):
            trigram = lowered[index : index + 3]
            digest = hashlib.sha256(trigram.encode()).digest()

            bucket = struct.unpack("<I", digest[:4])[0] % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            buckets[bucket] += sign

        # A zero vector would make cosine similarity undefined.
        if not any(buckets):
            buckets[0] = 1.0

        return normalise(buckets)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_provider(
    *,
    provider: str,
    api_key: str | None,
    model: str,
    dimensions: int,
    batch_size: int,
    requests_per_minute: int = 0,
) -> EmbeddingProvider:
    """Selects a provider from configuration."""
    if provider == "fake":
        return FakeEmbeddingProvider(dimensions=dimensions)

    if provider == "gemini":
        if not api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY")

        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            requests_per_minute=requests_per_minute,
        )

    raise EmbeddingError(f"unknown embedding provider: {provider}")
