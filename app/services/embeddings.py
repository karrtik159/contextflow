"""
Canonical embedding service — single source of truth for all vector generation.

Supports two providers (set via EMBEDDING_PROVIDER):
  - "openai"      → AsyncOpenAI / OpenAI embeddings API
  - "huggingface" → Local SentenceTransformer model

All embedding callers (semantic cache, pgvector search, CrewAI tools, context
prefetch) MUST use this module instead of ad-hoc client construction.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Thread-safe SentenceTransformer Singleton ────────────────

_hf_encoder: SentenceTransformer | None = None
_hf_encoder_lock = threading.Lock()


def _get_hf_encoder() -> SentenceTransformer:
    """Thread-safe lazy loader for the SentenceTransformer model.

    Uses a lock to prevent race conditions if multiple asyncio threads
    call this before the model is loaded. SentenceTransformer's internal
    tokenizer is NOT thread-safe for concurrent initialization.
    """
    global _hf_encoder
    if _hf_encoder is not None:
        return _hf_encoder

    with _hf_encoder_lock:
        # Double-checked locking: another thread may have loaded it while we waited
        if _hf_encoder is not None:
            return _hf_encoder

        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = settings.EMBEDDING_MODEL

        logger.info("Loading SentenceTransformer model '%s' on device '%s'...", model_name, device)
        _hf_encoder = SentenceTransformer(model_name, device=device)
        logger.info(
            "SentenceTransformer ready — dim=%d, device=%s",
            _hf_encoder.get_sentence_embedding_dimension(),
            device,
        )
    return _hf_encoder


def init_local_embedding_model() -> None:
    """Pre-load the embedding model into memory during FastAPI lifespan startup.

    Called synchronously from the lifespan factory. Validates that the loaded
    model dimension matches EMBEDDING_DIMENSIONS in config to catch .env
    mismatches early (before the first request hits Neo4j).
    """
    if settings.EMBEDDING_PROVIDER != "huggingface":
        return

    try:
        encoder = _get_hf_encoder()
        actual_dim = encoder.get_sentence_embedding_dimension()
        expected_dim = settings.EMBEDDING_DIMENSIONS

        if actual_dim != expected_dim:
            logger.error(
                "EMBEDDING_DIMENSIONS mismatch! Model '%s' produces %d-dim vectors, "
                "but config says %d. Update EMBEDDING_DIMENSIONS in .env to %d.",
                settings.EMBEDDING_MODEL, actual_dim, expected_dim, actual_dim,
            )
        else:
            logger.info("Embedding model validated — %d dimensions match config.", actual_dim)
    except Exception as exc:
        logger.error("Failed to load local embedding model: %s", exc, exc_info=True)


# ── Embedding Client Factory ────────────────────────────────


def _get_openai_embedding_client():
    """Build a sync OpenAI client for the configured embedding provider."""
    from openai import OpenAI

    from app.services.llm_provider import get_openai_compatible_kwargs

    provider = settings.EMBEDDING_PROVIDER

    if provider == "google":
        return OpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # "openai" and any other OpenAI-compatible provider
    return OpenAI(**get_openai_compatible_kwargs())


def _get_async_openai_embedding_client():
    """Build an async OpenAI client for the configured embedding provider."""
    from openai import AsyncOpenAI

    from app.services.llm_provider import get_openai_compatible_kwargs

    provider = settings.EMBEDDING_PROVIDER

    if provider == "google":
        return AsyncOpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # "openai" and any other OpenAI-compatible provider
    return AsyncOpenAI(**get_openai_compatible_kwargs())


# ── Dimension Validation ────────────────────────────────────


def _validate_embedding_dimensions(vector: list[float]) -> list[float]:
    if len(vector) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"model returned {len(vector)} dims but the application is configured for "
            f"{settings.EMBEDDING_DIMENSIONS}. Re-embed data and migrate pgvector if you switch dimensions."
        )
    return vector


# ── Embedding API kwargs ────────────────────────────────────


def _build_embedding_kwargs(text: str) -> dict:
    """Build the kwargs dict for an OpenAI embeddings.create() call."""
    kwargs = {
        "model": settings.EMBEDDING_MODEL,
        "input": text,
    }
    # Only explicitly pass dimensions for official OpenAI text-embedding-3 series
    # to prevent crashing local instances like LM Studio.
    if settings.EMBEDDING_PROVIDER == "openai" and "text-embedding-3" in settings.EMBEDDING_MODEL:
        kwargs["dimensions"] = settings.EMBEDDING_DIMENSIONS
    return kwargs


# ── Public API ──────────────────────────────────────────────


def embed_text(text: str) -> list[float]:
    """Synchronous embedding — used by CrewAI tools (which run in threads).

    Returns a validated, L2-normalized float vector.
    """
    if settings.EMBEDDING_PROVIDER == "huggingface":
        encoder = _get_hf_encoder()
        vec = encoder.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _validate_embedding_dimensions(vec.tolist())

    client = _get_openai_embedding_client()
    response = client.embeddings.create(**_build_embedding_kwargs(text))
    return _validate_embedding_dimensions(response.data[0].embedding)


async def embed_text_async(text: str) -> list[float]:
    """Async embedding — used by FastAPI endpoints.

    Returns a validated, L2-normalized float vector.
    """
    if settings.EMBEDDING_PROVIDER == "huggingface":
        # Offload to a thread to keep the FastAPI event loop free.
        # NOTE: SentenceTransformer.encode() holds the GIL during tokenization
        # but releases it during the PyTorch forward pass, so this does help
        # with concurrent request handling.
        return await asyncio.to_thread(embed_text, text)

    client = _get_async_openai_embedding_client()
    response = await client.embeddings.create(**_build_embedding_kwargs(text))
    return _validate_embedding_dimensions(response.data[0].embedding)


async def embed_text_async_safe(text: str) -> list[float] | None:
    """Like embed_text_async but returns None on failure instead of raising.

    Use this for best-effort flows like semantic caching where a missing
    embedding should gracefully degrade rather than crash the request.
    """
    try:
        return await embed_text_async(text)
    except Exception as exc:
        logger.warning("Embedding generation failed (%s), skipping", exc)
        return None
