"""
Integration test for the RAG query endpoint.

NOTE: This test requires the full app stack (PostgreSQL, asyncpg) and runs
in Docker/CI environments. It will NOT work in local development without
the database running.

The test mocks the SupportCrew, intent classifier, and embedding service
to isolate the HTTP transport + routing logic.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_optional_user
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


async def _auth_user():
    return {"id": "user-123", "username": "user-123"}


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
        "app.services.embeddings.embed_text_async_safe",
        _no_embedding,
    )
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *args, **kwargs: None)

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
        "app.services.embeddings.embed_text_async_safe",
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


@pytest.mark.asyncio
async def test_rag_query_rejects_spoofed_user_id(monkeypatch):
    app.dependency_overrides[get_optional_user] = _auth_user
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "What did I decide?", "user_id": "other-user"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_rag_query_service_token_can_supply_user_id(monkeypatch):
    monkeypatch.setattr(
        "app.api.deps.settings.RAG_SERVICE_TOKEN",
        type("Secret", (), {"get_secret_value": lambda self: "service-secret"})(),
    )
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        _no_embedding,
    )
    monkeypatch.setattr("app.api.v1.rag._process_memory_background", lambda *args, **kwargs: None)

    class FakeCrewRunner:
        def kickoff(self, *, inputs):
            assert inputs["user_id"] == "user-123"
            return "Scoped answer from the knowledge service."

    class FakeSupportCrew:
        def crew(self):
            return FakeCrewRunner()

    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", FakeSupportCrew)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            headers={"X-RAG-Service-Token": "service-secret"},
            json={"query": "What did I decide?", "user_id": "user-123"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-123"
    assert data["routed_to"] == "crewai"


@pytest.mark.asyncio
async def test_rag_query_anonymous_does_not_use_semantic_cache(monkeypatch):
    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", _FakeSupportCrew)
    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )

    async def fail_embedding(*args, **kwargs):
        raise AssertionError("anonymous RAG requests must not generate cache embeddings")

    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        fail_embedding,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "How does the worker reach the backend?"},
        )

    assert response.status_code == 200
    assert response.json()["routed_to"] == "crewai"


@pytest.mark.asyncio
async def test_rag_query_authenticated_cache_is_user_scoped(monkeypatch):
    captured: dict[str, object] = {}
    app.dependency_overrides[get_optional_user] = _auth_user

    async def embedding(*args, **kwargs):
        return [0.1, 0.2, 0.3]

    async def cached_response(*, normalized_query, embedding, user_id):
        captured["normalized_query"] = normalized_query
        captured["embedding"] = embedding
        captured["user_id"] = user_id
        return "Cached scoped answer"

    monkeypatch.setattr(
        "app.services.llm_provider.classify_intent",
        _needs_rag,
    )
    monkeypatch.setattr(
        "app.services.embeddings.embed_text_async_safe",
        embedding,
    )
    monkeypatch.setattr(
        "app.services.semantic_cache.get_cached_response",
        cached_response,
    )

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/rag/query",
                json={"query": "What did I decide?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Cached scoped answer"
    assert data["user_id"] == "user-123"
    assert data["routed_to"] == "cache"
    assert captured["user_id"] == "user-123"
