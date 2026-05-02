"""
Multi-provider LLM configuration for CrewAI.

Supported providers (set via LLM_PROVIDER env var):
  - "openai"     → Direct OpenAI API (or any OpenAI-compatible endpoint)
  - "google"     → Google Gemini via native CrewAI support
  - "openrouter" → OpenRouter via LiteLLM prefix (openrouter/model-name)

Usage:
    from app.services.llm_provider import build_crewai_llm
    llm = build_crewai_llm()
"""

from __future__ import annotations

import threading

from crewai import LLM

from app.core.config import settings


def get_openai_compatible_kwargs() -> dict:
    """Build kwargs for any OpenAI-compatible client (embeddings, direct calls)."""
    kwargs: dict = {}
    api_key = settings.OPENAI_API_KEY.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return kwargs


def build_crewai_llm() -> LLM:
    """Build a CrewAI-compatible LLM based on the configured provider."""
    provider = settings.LLM_PROVIDER.lower().strip()
    model = settings.LLM_MODEL

    if provider == "openai":
        # Native OpenAI — pass model as-is (e.g. "gpt-4.1-mini")
        return LLM(
            model=model,
            temperature=0,
            **get_openai_compatible_kwargs(),
        )

    if provider == "google":
        # Native Google Gemini — CrewAI recognizes "gemini/" prefix
        prefixed = model if model.startswith("gemini/") else f"gemini/{model}"
        return LLM(
            model=prefixed,
            temperature=0,
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        )

    if provider == "openrouter":
        # OpenRouter via LiteLLM — needs "openrouter/" prefix
        prefixed = model if model.startswith("openrouter/") else f"openrouter/{model}"
        return LLM(
            model=prefixed,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Supported: 'openai', 'google', 'openrouter'")


def build_crewai_embedder() -> dict:
    """Build the embedder config dict for CrewAI's internal memory system.

    CrewAI's ``Crew(memory=True)`` defaults to OpenAI embeddings.
    This function returns an ``embedder`` dict that matches the
    configured EMBEDDING_PROVIDER so CrewAI's memory works with
    whichever embedding backend is active.

    Usage:
        Crew(memory=True, embedder=build_crewai_embedder())
    """
    provider = settings.EMBEDDING_PROVIDER.lower().strip()

    if provider == "openai":
        config: dict = {
            "model": settings.EMBEDDING_MODEL,
        }
        api_key = settings.OPENAI_API_KEY.get_secret_value()
        if api_key:
            config["api_key"] = api_key
        if settings.OPENAI_BASE_URL:
            config["api_base"] = settings.OPENAI_BASE_URL
        return {"provider": "openai", "config": config}

    if provider == "huggingface":
        return {
            "provider": "huggingface",
            "config": {"model": settings.EMBEDDING_MODEL},
        }
    
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER for CrewAI memory: '{provider}'. "
        f"Supported: 'openai', 'huggingface'"
    )


# ── Direct LLM Client (bypasses CrewAI) ─────────────────────


def get_async_llm_client():
    """Return an AsyncOpenAI client configured for the active LLM_PROVIDER.

    All three providers (openai, google, openrouter) support the
    OpenAI-compatible chat completions API, so we use the same
    ``openai.AsyncOpenAI`` client for all of them.

    - **openai**: Uses OPENAI_API_KEY + optional OPENAI_BASE_URL.
    - **google**: Uses the Gemini OpenAI-compat endpoint at
      ``generativelanguage.googleapis.com/v1beta/openai/``.
    - **openrouter**: Uses OPENAI_API_KEY + ``openrouter.ai/api/v1``.
    """
    from openai import AsyncOpenAI

    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "openai":
        return AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or None,
        )

    if provider == "google":
        return AsyncOpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if provider == "openrouter":
        return AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'")


def _get_model_name() -> str:
    """Return the model name suitable for the OpenAI-compatible client."""
    provider = settings.LLM_PROVIDER.lower().strip()
    model = settings.LLM_MODEL

    if provider == "google":
        # Gemini OpenAI-compat expects bare model name (no "gemini/" prefix)
        return model.removeprefix("gemini/")

    # openai / openrouter: model name as-is
    return model


async def classify_intent(query: str) -> bool:
    """Fast intent classification — returns True if the query needs RAG.

    Uses a two-layer strategy:
    1. **Keyword heuristic** — catches obvious greetings/small-talk in 0ms.
    2. **LLM classification** — for ambiguous queries, uses a lightweight
       LLM call (~200ms) with text output (no JSON mode for provider compat).

    Returns False for simple chat, True for knowledge queries.
    """
    import logging

    logger = logging.getLogger(__name__)

    # ── Layer 1: Fast keyword heuristic (0ms) ──────────────
    query_lower = query.lower().strip()
    query_words = query_lower.split()

    # Obvious greetings / small talk — never need RAG
    SIMPLE_PATTERNS = {
        "hi", "hello", "hey", "howdy", "yo", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see ya", "later",
        "good morning", "good evening", "good night",
        "how are you", "what's up", "whats up",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
        "lol", "haha", "nice", "cool", "great", "awesome",
    }

    GREETING_FIRST_WORDS = {
        "hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yo", "sup",
    }

    if query_lower in SIMPLE_PATTERNS or (
        query_words
        and len(query_words) <= 3
        and query_words[0] in GREETING_FIRST_WORDS
    ):
        logger.info("Intent [heuristic]: simple chat — '%s'", query[:60])
        return False

    # ── Layer 2: LLM classification (ambiguous queries) ────
    client = get_async_llm_client()
    model = _get_model_name()

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. "
                        "Reply with EXACTLY one word: RAG or CHAT.\n"
                        "Say RAG if the user needs factual info, document lookup, "
                        "or knowledge retrieval.\n"
                        "Say CHAT if it's a greeting, small talk, opinion, "
                        "thanks, or casual conversation."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content.strip().upper()
        needs_rag = "RAG" in answer
        logger.info(
            "Intent [LLM]: %s — '%s' (raw: %s)",
            "RAG" if needs_rag else "CHAT",
            query[:60],
            answer,
        )
        return needs_rag
    except Exception as exc:
        logger.warning("Intent classification failed (%s), defaulting to RAG", exc)
        return True


_hf_encoder = None
_hf_encoder_lock = threading.Lock()


def _get_hf_encoder():
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

        import logging

        import torch
        from sentence_transformers import SentenceTransformer

        logger = logging.getLogger(__name__)

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


def init_local_embedding_model():
    """Pre-load the embedding model into memory during FastAPI lifespan startup.

    Called synchronously from the lifespan factory. Validates that the loaded
    model dimension matches EMBEDDING_DIMENSIONS in config to catch .env
    mismatches early (before the first request hits Neo4j).
    """
    import logging

    logger = logging.getLogger(__name__)

    if settings.EMBEDDING_PROVIDER.lower().strip() != "huggingface":
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


async def _get_embedding(text: str) -> list[float] | None:
    """Generate a vector embedding for the given text.

    Uses the configured EMBEDDING_PROVIDER / EMBEDDING_MODEL to produce
    a float vector suitable for Neo4j semantic cache lookups.

    Returns None on failure so the caller can gracefully skip caching.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        provider = settings.EMBEDDING_PROVIDER.lower().strip()

        if provider == "huggingface":
            import asyncio
            
            def _encode_sync(content: str) -> list[float]:
                encoder = _get_hf_encoder()
                # normalize_embeddings=True: L2-normalizes vectors so cosine similarity
                # reduces to a fast dot product inside Neo4j's vector index.
                # show_progress_bar=False: prevents stdout pollution in server logs.
                vec = encoder.encode(
                    content,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return vec.tolist()
                
            # Offload to a thread to keep the FastAPI event loop free.
            # NOTE: SentenceTransformer.encode() holds the GIL during tokenization
            # but releases it during the PyTorch forward pass, so this does help
            # with concurrent request handling.
            return await asyncio.to_thread(_encode_sync, text)

        else:
            from openai import AsyncOpenAI
            
            if provider == "openai":
                client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY.get_secret_value(),
                    base_url=settings.OPENAI_BASE_URL or None,
                )
            elif provider == "google":
                # Gemini exposes an OpenAI-compatible embeddings endpoint
                client = AsyncOpenAI(
                    api_key=settings.GOOGLE_API_KEY.get_secret_value(),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
            else:
                client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY.get_secret_value(),
                    base_url=settings.OPENAI_BASE_URL or None,
                )
    
            kwargs = {
                "model": settings.EMBEDDING_MODEL,
                "input": text,
            }
            
            # Only explicitly pass dimensions if it's the official OpenAI Provider using text-embedding-3 series 
            # to prevent crashing local instances like LM Studio.
            if provider == "openai" and "text-embedding-3" in settings.EMBEDDING_MODEL:
                kwargs["dimensions"] = settings.EMBEDDING_DIMENSIONS
    
            response = await client.embeddings.create(**kwargs)
            return response.data[0].embedding

    except Exception as exc:
        logger.warning("Embedding generation failed (%s), skipping cache", exc)
        return None


async def stream_direct_chat(
    query: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
):
    """Stream a direct LLM response, bypassing CrewAI entirely.

    Yields content chunks as they arrive. Use for simple chat queries
    that don't need RAG tool invocation.

    Args:
        query: The user's message.
        system_prompt: Optional system prompt override.
        history: Optional list of prior messages [{"role": ..., "content": ...}].

    Yields:
        str: Content delta chunks from the streaming response.
    """
    client = get_async_llm_client()
    model = _get_model_name()

    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful, friendly AI assistant. "
                "Keep answers concise and conversational."
            ),
        })

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": query})

    stream = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        stream=True,
        messages=messages,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def direct_chat(
    query: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> str:
    """Non-streaming direct LLM call. Returns the full response string."""
    chunks: list[str] = []
    async for chunk in stream_direct_chat(
        query, system_prompt=system_prompt, history=history
    ):
        chunks.append(chunk)
    return "".join(chunks)
