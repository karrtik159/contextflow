"""
Integration test for the RAG query endpoint.

NOTE: This test requires the full app stack (PostgreSQL, asyncpg) and runs
in Docker/CI environments. It will NOT work in local development without
the database running.

The test mocks the SupportCrew, intent classifier, and embedding service
to isolate the HTTP transport + routing logic.
"""

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


async def _needs_rag(*args, **kwargs):
    return True


async def _simple_chat(*args, **kwargs):
    return False


async def _no_embedding(*args, **kwargs):
    return None


async def _direct_answer(*args, **kwargs):
    return "Hey there!"


class _FakeCrewRunner:
    def kickoff(self, *, inputs):
        assert inputs["query"] == "How does the worker reach the backend?"
        assert inputs["user_id"] == "anonymous"
        return "The worker calls the FastAPI RAG endpoint over HTTP."


class _FakeSupportCrew:
    def crew(self):
        return _FakeCrewRunner()


@pytest.mark.asyncio
async def test_rag_query_crewai_path(monkeypatch):
    """Full HTTP roundtrip through the RAG endpoint — crew path."""
    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", _FakeSupportCrew)
    # Force intent classifier to say "needs RAG"
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    # Skip embedding + cache
    monkeypatch.setattr(
        "app.services.llm_provider._get_embedding",
        _no_embedding,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "How does the worker reach the backend?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The worker calls the FastAPI RAG endpoint over HTTP."
    assert data["query"] == "How does the worker reach the backend?"
    assert data["user_id"] is None
    assert data["routed_to"] == "crewai"


@pytest.mark.asyncio
async def test_rag_query_direct_path(monkeypatch):
    """Full HTTP roundtrip — direct chat path (bypasses CrewAI)."""
    # Force intent classifier to say "simple chat"
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _simple_chat,
    )
    # Mock direct_chat
    monkeypatch.setattr(
        "app.services.llm_provider.direct_chat",
        _direct_answer,
    )
    # Skip embedding + cache
    monkeypatch.setattr(
        "app.services.llm_provider._get_embedding",
        _no_embedding,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "Hello!"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hey there!"
    assert data["routed_to"] == "direct"
