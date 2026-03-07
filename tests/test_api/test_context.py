from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.v1.context import PrefetchRequest, _parse_uuid, prefetch_context


class _FakeEmbeddingsClient:
    async def create(self, input: str, model: str):
        assert input == "where did we leave off?"
        assert model == "text-embedding-3-small"
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeOpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = _FakeEmbeddingsClient()


@pytest.mark.asyncio
async def test_prefetch_context_scopes_results_by_user_and_session(monkeypatch):
    captured: dict[str, object] = {}
    user_id = uuid4()
    session_id = uuid4()

    async def fake_search_similar_messages(*, db, query_embedding, session_id=None, user_id=None, limit=5):
        captured["db"] = db
        captured["query_embedding"] = query_embedding
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        captured["limit"] = limit
        return [
            SimpleNamespace(role="user", content="You asked for the deployment checklist."),
            SimpleNamespace(role="assistant", content="We paused at the LiveKit worker setup."),
        ]

    monkeypatch.setattr("app.api.v1.context.search_similar_messages", fake_search_similar_messages)
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAIClient)

    response = await prefetch_context(
        PrefetchRequest(
            query="where did we leave off?",
            user_id=user_id,
            session_id=session_id,
            limit=3,
        ),
        db=object(),
    )

    assert "Relevant context from previous memory/conversations" in response["context"]
    assert "[USER]: You asked for the deployment checklist." in response["context"]
    assert captured == {
        "db": captured["db"],
        "query_embedding": [0.1, 0.2, 0.3],
        "session_id": session_id,
        "user_id": user_id,
        "limit": 3,
    }


@pytest.mark.asyncio
async def test_prefetch_context_requires_scoped_identifiers(monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAIClient)

    response = await prefetch_context(
        PrefetchRequest(
            query="where did we leave off?",
            user_id="anonymous",
            session_id=None,
            limit=5,
        ),
        db=object(),
    )

    assert response == {"context": "No scoped conversation context is available yet."}


def test_parse_uuid_handles_invalid_values():
    assert _parse_uuid(None) is None
    assert _parse_uuid("") is None
    assert _parse_uuid("anonymous") is None
    assert _parse_uuid("not-a-uuid") is None

    value = uuid4()
    assert _parse_uuid(value) == value
    assert _parse_uuid(str(value)) == UUID(str(value))
