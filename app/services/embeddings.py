"""
Embedding provider abstraction for OpenAI-compatible and Hugging Face models.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI

from app.core.config import settings
from app.services.llm_provider import get_openai_compatible_kwargs


def _validate_embedding_dimensions(vector: list[float]) -> list[float]:
    if len(vector) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"model returned {len(vector)} dims but the application is configured for "
            f"{settings.EMBEDDING_DIMENSIONS}. Re-embed data and migrate pgvector if you switch dimensions."
        )
    return vector


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    if settings.EMBEDDING_PROVIDER == "openai":
        client = OpenAI(**get_openai_compatible_kwargs())
        response = client.embeddings.create(input=text, model=settings.EMBEDDING_MODEL)
        return _validate_embedding_dimensions(response.data[0].embedding)

    if settings.EMBEDDING_PROVIDER == "huggingface":
        vector = _get_sentence_transformer().encode(text, convert_to_numpy=True).tolist()
        return _validate_embedding_dimensions(vector)

    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")


async def embed_text_async(text: str) -> list[float]:
    if settings.EMBEDDING_PROVIDER == "openai":
        client = AsyncOpenAI(**get_openai_compatible_kwargs())
        response = await client.embeddings.create(input=text, model=settings.EMBEDDING_MODEL)
        return _validate_embedding_dimensions(response.data[0].embedding)

    return await asyncio.to_thread(embed_text, text)
