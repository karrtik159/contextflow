import pytest

from app.services.semantic_cache import get_cached_response, populate_semantic_cache


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _FakeSession:
    def __init__(self, records):
        self.records = list(records)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        record = self.records.pop(0) if self.records else None
        return _FakeResult(record)


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


@pytest.mark.asyncio
async def test_get_cached_response_exact_match_is_user_scoped(monkeypatch):
    session = _FakeSession(records=[{"answer": "Exact scoped answer"}])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    answer = await get_cached_response(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        user_id="user-a",
    )

    assert answer == "Exact scoped answer"
    assert len(session.calls) == 1
    assert "user_id: $user_id" in session.calls[0]["query"]
    assert session.calls[0]["params"]["user_id"] == "user-a"


@pytest.mark.asyncio
async def test_get_cached_response_vector_match_filters_by_user(monkeypatch):
    session = _FakeSession(records=[None, {"answer": "Vector scoped answer"}])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    answer = await get_cached_response(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        user_id="user-b",
    )

    assert answer == "Vector scoped answer"
    assert len(session.calls) == 2
    vector_call = session.calls[1]
    assert "c.user_id = $user_id" in vector_call["query"]
    assert vector_call["params"]["user_id"] == "user-b"
    assert vector_call["params"]["candidate_count"] == 50


@pytest.mark.asyncio
async def test_populate_semantic_cache_upserts_per_user_query(monkeypatch):
    session = _FakeSession(records=[None])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    await populate_semantic_cache(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        answer="Scoped answer",
        user_id="user-c",
        session_id="session-1",
    )

    assert len(session.calls) == 1
    call = session.calls[0]
    assert "MERGE (c:SemanticCache" in call["query"]
    assert "CREATE (c:SemanticCache" not in call["query"]
    assert call["params"]["user_id"] == "user-c"
    assert call["params"]["session_id"] == "session-1"
