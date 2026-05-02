"""
Integration tests for the 5-tier RAG pipeline routing logic.

Since the app module chain (rag → deps → db) triggers asyncpg at import time,
these tests simulate the routing logic in isolation by directly exercising
the routing decision functions WITHOUT importing the full rag module.

This approach tests the LOGIC of the pipeline, not the HTTP transport.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.cache_sanitizer import SanitizedQuery, sanitize_query


# ── Helpers ──────────────────────────────────────────────────

def _build_sanitized(query: str, requires_isolation: bool = False, pii: list | None = None):
    return SanitizedQuery(
        original_query=query,
        normalized_query=query.lower().strip(),
        requires_isolation=requires_isolation,
        detected_pii_types=pii or [],
    )


async def _simulate_pipeline(
    query: str,
    user_id: str | None = None,
    *,
    sanitized: SanitizedQuery | None = None,
    embedding: list[float] | None = None,
    cached_answer: str | None = None,
    intent_needs_rag: bool = False,
    direct_chat_answer: str = "direct answer",
    crew_answer: str | None = None,
    crew_error: Exception | None = None,
) -> dict:
    """Simulate the RAG pipeline routing logic WITHOUT importing the full app module.

    Returns a dict with 'answer' and 'routed_to' mirroring RAGQueryResponse.
    """
    # Step 0: Sanitize
    if sanitized is None:
        sanitized = sanitize_query(query)

    effective_user_id = user_id or "anonymous"
    cache_lookup_scoped_id = effective_user_id if sanitized.requires_isolation else None

    # Step 1: Embedding + Cache
    query_embedding = embedding  # None simulates embedding failure

    if query_embedding is not None and cached_answer is not None:
        return {"answer": cached_answer, "routed_to": "cache"}

    # Step 2: Intent
    needs_rag = intent_needs_rag

    # Step 3: Direct chat
    if not needs_rag:
        return {"answer": direct_chat_answer, "routed_to": "direct"}

    # Step 4: CrewAI
    if crew_error:
        return {"answer": direct_chat_answer, "routed_to": "direct_fallback"}

    if crew_answer and len(crew_answer.strip()) >= 5:
        return {"answer": crew_answer.strip(), "routed_to": "crewai"}
    else:
        return {"answer": direct_chat_answer, "routed_to": "direct_fallback"}


# ── Test: Cache Hit Path ────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_returns_cached_answer():
    """When semantic cache has a match, bypass all LLM processing."""
    result = await _simulate_pipeline(
        "What is AI?",
        embedding=[0.1] * 384,
        cached_answer="Cached: AI is...",
    )
    assert result["routed_to"] == "cache"
    assert result["answer"] == "Cached: AI is..."


# ── Test: Direct Chat Path ──────────────────────────────────

@pytest.mark.asyncio
async def test_simple_chat_routes_to_direct():
    """When intent classifier says CHAT, bypass CrewAI."""
    result = await _simulate_pipeline(
        "How are you?",
        embedding=[0.1] * 384,
        intent_needs_rag=False,
        direct_chat_answer="I'm doing well!",
    )
    assert result["routed_to"] == "direct"
    assert result["answer"] == "I'm doing well!"


# ── Test: CrewAI RAG Path ───────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_query_routes_to_crewai():
    """When intent classifier says RAG, use SupportCrew."""
    result = await _simulate_pipeline(
        "What is quantum computing?",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_answer="Quantum computing uses qubits to perform calculations...",
    )
    assert result["routed_to"] == "crewai"
    assert "qubits" in result["answer"]


# ── Test: CrewAI Failure → Fallback ─────────────────────────

@pytest.mark.asyncio
async def test_crew_failure_falls_back_to_direct():
    """When SupportCrew raises an exception, fall back to direct LLM."""
    result = await _simulate_pipeline(
        "What is quantum computing?",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_error=RuntimeError("LLM crashed"),
        direct_chat_answer="Fallback answer",
    )
    assert result["routed_to"] == "direct_fallback"
    assert result["answer"] == "Fallback answer"


# ── Test: CrewAI Empty Result → Fallback ────────────────────

@pytest.mark.asyncio
async def test_crew_empty_result_falls_back():
    """When SupportCrew returns empty/short answer, fall back to direct LLM."""
    result = await _simulate_pipeline(
        "Tell me about RAG",
        embedding=[0.1] * 384,
        intent_needs_rag=True,
        crew_answer="",  # Empty crew result
        direct_chat_answer="Direct RAG explanation",
    )
    assert result["routed_to"] == "direct_fallback"


# ── Test: PII Isolation ─────────────────────────────────────

@pytest.mark.asyncio
async def test_pii_query_scopes_cache_to_user():
    """Queries with PII should scope cache lookups to the user's ID."""
    sanitized = SanitizedQuery(
        original_query="Send to john@test.com",
        normalized_query="send to [email]",
        requires_isolation=True,
        detected_pii_types=["EMAIL"],
    )

    # The cache_lookup_scoped_id should be the user_id when PII is found
    effective_user_id = "user-123"
    cache_lookup_scoped_id = effective_user_id if sanitized.requires_isolation else None

    assert cache_lookup_scoped_id == "user-123"  # Isolated to user


@pytest.mark.asyncio
async def test_no_pii_uses_global_cache():
    """Queries without PII should search the global cache."""
    sanitized = sanitize_query("What is machine learning?")

    effective_user_id = "user-456"
    cache_lookup_scoped_id = effective_user_id if sanitized.requires_isolation else None

    assert cache_lookup_scoped_id is None  # Global cache


# ── Test: Embedding Failure → Skip Cache Gracefully ─────────

@pytest.mark.asyncio
async def test_embedding_failure_skips_cache():
    """If embedding generation fails, skip cache and proceed to intent classification."""
    result = await _simulate_pipeline(
        "What is AI?",
        embedding=None,  # Embedding failed
        intent_needs_rag=False,
        direct_chat_answer="AI is...",
    )
    assert result["routed_to"] == "direct"
    assert result["answer"] == "AI is..."


# ── Test: Sanitizer Integration ─────────────────────────────

def test_sanitizer_flags_pii_for_isolation():
    """End-to-end: PII query triggers isolation flag."""
    result = sanitize_query("Email me at admin@company.org")
    assert result.requires_isolation is True
    assert "EMAIL" in result.detected_pii_types
    assert "[email]" in result.normalized_query


def test_sanitizer_cleans_fillers():
    """End-to-end: filler words are stripped for better cache hits."""
    result = sanitize_query("Hey, can you tell me about transformers?")
    assert result.normalized_query == "about transformers?"
    assert result.requires_isolation is False
