from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class _FakeCrewRunner:
    def kickoff(self, *, inputs):
        assert inputs == {"query": "How does the worker reach the backend?", "user_id": "anonymous"}
        return "The worker calls the FastAPI RAG endpoint over HTTP."


class _FakeSupportCrew:
    def crew(self):
        return _FakeCrewRunner()


@pytest.mark.asyncio
async def test_rag_query_returns_support_crew_answer(monkeypatch):
    monkeypatch.setattr("agents.crews.support_crew.SupportCrew", _FakeSupportCrew)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"query": "How does the worker reach the backend?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The worker calls the FastAPI RAG endpoint over HTTP.",
        "query": "How does the worker reach the backend?",
        "user_id": None,
    }
