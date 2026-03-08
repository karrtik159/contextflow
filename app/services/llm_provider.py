"""
Helpers for configuring LLM providers across CrewAI, RAGAS, and SDK calls.
"""

from __future__ import annotations

from crewai import LLM

from app.core.config import settings


def get_openai_compatible_kwargs() -> dict:
    kwargs: dict = {}
    api_key = settings.OPENAI_API_KEY.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return kwargs


def build_crewai_llm() -> LLM:
    return LLM(
        model=settings.LLM_MODEL,
        temperature=0,
        **get_openai_compatible_kwargs(),
    )
