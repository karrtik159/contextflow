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
