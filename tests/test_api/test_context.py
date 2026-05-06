from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.context import PrefetchRequest, _parse_uuid, prefetch_context


class _FakeEmbeddingsClient:
    async def __call__(self, text: str):
        assert text == "where did we leave off?"
        return [0.1, 0.2, 0.3]


def _request(headers: dict[str, str] | None = None):
    return SimpleNamespace(headers=headers or {})


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
    fake_embedder = _FakeEmbeddingsClient()
    monkeypatch.setattr("app.api.v1.context.embed_text_async", fake_embedder)

    response = await prefetch_context(
        PrefetchRequest(
            query="where did we leave off?",
            user_id=user_id,
            session_id=session_id,
            limit=3,
        ),
        db=object(),
        http_request=_request(),
        user={"id": str(user_id)},
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
    monkeypatch.setattr(
        "app.api.deps.settings.RAG_SERVICE_TOKEN",
        type("Secret", (), {"get_secret_value": lambda self: "service-secret"})(),
    )

    response = await prefetch_context(
        PrefetchRequest(
            query="where did we leave off?",
            user_id="anonymous",
            session_id=None,
            limit=5,
        ),
        db=object(),
        http_request=_request({"X-RAG-Service-Token": "service-secret"}),
        user=None,
    )

    assert response == {"context": "No scoped conversation context is available yet."}


@pytest.mark.asyncio
async def test_prefetch_context_rejects_unauthenticated_request():
    with pytest.raises(HTTPException) as exc:
        await prefetch_context(
            PrefetchRequest(
                query="where did we leave off?",
                user_id=None,
                session_id=None,
                limit=5,
            ),
            db=object(),
            http_request=_request(),
            user=None,
        )

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_prefetch_context_rejects_spoofed_user_id():
    with pytest.raises(HTTPException) as exc:
        await prefetch_context(
            PrefetchRequest(
                query="where did we leave off?",
                user_id=uuid4(),
                session_id=None,
                limit=5,
            ),
            db=object(),
            http_request=_request(),
            user={"id": str(uuid4())},
        )

    assert exc.value.status_code == 403


def test_parse_uuid_handles_invalid_values():
    assert _parse_uuid(None) is None
    assert _parse_uuid("") is None
    assert _parse_uuid("anonymous") is None
    assert _parse_uuid("not-a-uuid") is None

    value = uuid4()
    assert _parse_uuid(value) == value
    assert _parse_uuid(str(value)) == UUID(str(value))
