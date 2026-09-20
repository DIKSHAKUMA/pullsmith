"""Probes the embedding provider: does the key work, and what dimension does it return?

Run once when setting up a key. It answers three questions that everything else depends
on, and answers them cheaply rather than discovering them halfway through indexing a
repository:

1. Does the credential authenticate at all?
2. Which embedding models are available to this key?
3. What vector dimension do they return? (pgvector's HNSW index caps at 2000, and the
   column dimension is fixed at migration time, so this must be known first.)

The key is read from the environment and never printed.

Usage:  python -m scripts.probe_embeddings
"""

import asyncio
import os

import httpx

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

CANDIDATE_MODELS = (
    "gemini-embedding-001",
    "text-embedding-004",
    "embedding-001",
)


async def list_models(client: httpx.AsyncClient, key: str) -> list[str]:
    response = await client.get(
        f"{GEMINI_BASE}/models",
        headers={"x-goog-api-key": key},
    )

    print(f"GET /models -> {response.status_code}")

    if response.status_code != 200:
        # Body may name the reason (invalid key, API not enabled, wrong project).
        print(f"  {response.text[:400]}")
        return []

    payload = response.json()
    names = [model["name"].removeprefix("models/") for model in payload.get("models", [])]

    embedding_models = [
        name
        for name, model in zip(names, payload.get("models", []), strict=False)
        if any(
            "embed" in method.lower()
            for method in model.get("supportedGenerationMethods", [])
        )
    ]

    print(f"  models visible      : {len(names)}")
    print(f"  embedding-capable   : {embedding_models or '(none reported)'}")
    return names


async def try_embed(client: httpx.AsyncClient, key: str, model: str) -> int | None:
    """Embeds one short string and returns the vector dimension."""
    response = await client.post(
        f"{GEMINI_BASE}/models/{model}:embedContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "model": f"models/{model}",
            "content": {"parts": [{"text": "def validate_email(email): return '@' in email"}]},
        },
    )

    if response.status_code != 200:
        print(f"  {model:<24} -> {response.status_code} {response.text[:160]}")
        return None

    values = response.json().get("embedding", {}).get("values", [])
    print(f"  {model:<24} -> 200, dimension {len(values)}")
    return len(values)


async def main() -> None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment")

    print(f"key length {len(key)}, prefix {key[:3]}...\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        await list_models(client, key)

        print("\nembedding attempts:")
        for model in CANDIDATE_MODELS:
            dimension = await try_embed(client, key, model)
            if dimension:
                print(f"\nusable model: {model} at {dimension} dimensions")
                return

    print("\nno embedding model succeeded with this credential")


if __name__ == "__main__":
    asyncio.run(main())
