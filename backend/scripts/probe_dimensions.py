"""Checks reduced-dimension embeddings and batching.

Why this matters
----------------
``gemini-embedding-001`` returns 3072 dimensions, but pgvector's ``vector`` type only
supports HNSW indexing up to 2000 dimensions. Without an index every search becomes a
full scan of every chunk, which does not scale.

The model supports Matryoshka Representation Learning: the most important information is
concentrated in the leading dimensions, so the vector can be truncated with modest quality
loss. Truncated vectors are no longer unit length, so they must be renormalised before
cosine comparisons.

This script verifies:
1. ``outputDimensionality`` is honoured
2. whether returned vectors are unit length at each size
3. that batching works, and how much faster it is
4. that reduced dimensions still rank related code above unrelated code

Usage:  python -m scripts.probe_dimensions
"""

import asyncio
import math
import os
import time

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-embedding-001"

QUERY = "where is user email validation handled?"

RELATED = "def validate_email(email: str) -> bool:\n    return '@' in email and len(email) > 3"
UNRELATED = (
    "def calculate_shipping_cost(weight: float, zone: int) -> float:\n"
    "    return weight * zone * 1.5"
)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (norm(a) * norm(b))


async def embed(
    client: httpx.AsyncClient, key: str, text: str, dimensions: int | None = None
) -> list[float]:
    body: dict = {"model": f"models/{MODEL}", "content": {"parts": [{"text": text}]}}

    if dimensions:
        body["outputDimensionality"] = dimensions

    response = await client.post(
        f"{BASE}/models/{MODEL}:embedContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=body,
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


async def embed_batch(
    client: httpx.AsyncClient, key: str, texts: list[str], dimensions: int
) -> list[list[float]]:
    response = await client.post(
        f"{BASE}/models/{MODEL}:batchEmbedContents",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "requests": [
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": dimensions,
                }
                for text in texts
            ]
        },
    )
    response.raise_for_status()
    return [item["values"] for item in response.json()["embeddings"]]


async def main() -> None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("dimension support and vector length:")
        for dimensions in (None, 1536, 768):
            vector = await embed(client, key, RELATED, dimensions)
            label = "default" if dimensions is None else str(dimensions)
            print(f"  {label:<8} -> {len(vector):>5} dims, L2 norm {norm(vector):.4f}")

        print("\nsemantic ranking at 1536 dimensions:")
        query_vector = await embed(client, key, QUERY, 1536)
        related_vector = await embed(client, key, RELATED, 1536)
        unrelated_vector = await embed(client, key, UNRELATED, 1536)

        related_score = cosine(query_vector, related_vector)
        unrelated_score = cosine(query_vector, unrelated_vector)

        print(f"  query vs related email code    : {related_score:.4f}")
        print(f"  query vs unrelated shipping code: {unrelated_score:.4f}")
        print(
            "  ranking correct"
            if related_score > unrelated_score
            else "  RANKING WRONG - reduced dimensions unusable"
        )

        print("\nbatching:")
        texts = [f"def handler_{index}(request): return request.json()" for index in range(16)]

        start = time.perf_counter()
        for text in texts:
            await embed(client, key, text, 1536)
        sequential = time.perf_counter() - start

        start = time.perf_counter()
        vectors = await embed_batch(client, key, texts, 1536)
        batched = time.perf_counter() - start

        print(f"  16 texts one by one : {sequential:.2f}s")
        print(f"  16 texts batched    : {batched:.2f}s ({len(vectors)} vectors)")
        print(f"  speedup             : {sequential / max(batched, 0.001):.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
