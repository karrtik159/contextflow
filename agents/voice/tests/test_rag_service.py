import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

VOICE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(VOICE_SRC) not in sys.path:
    sys.path.insert(0, str(VOICE_SRC))

from rag_service import (  # noqa: E402
    CONTEXT_PREFETCH_PATH,
    RAG_QUERY_PATH,
    get_chat_session_id,
    get_session_user_id,
    inject_prefetched_context,
    prefetch_context,
    search_knowledge_base,
)


def test_session_helpers_read_userdata():
    session = SimpleNamespace(userdata=SimpleNamespace(user_id="user-123", chat_session_id="session-456"))

    assert get_session_user_id(session) == "user-123"
    assert get_chat_session_id(session) == "session-456"
    assert get_session_user_id(SimpleNamespace(userdata=None)) == "anonymous"
    assert get_chat_session_id(SimpleNamespace(userdata=None)) is None


def test_inject_prefetched_context_adds_system_message():
    calls: list[tuple[str, str]] = []

    class FakeTurnContext:
        def add_message(self, *, role: str, content: str) -> None:
            calls.append((role, content))

    inject_prefetched_context(FakeTurnContext(), "Previous answer about the worker deployment.")

    assert calls == [
        (
            "system",
            "Relevant context for the user's last message:\nPrevious answer about the worker deployment.",
        )
    ]


@pytest.mark.asyncio
async def test_prefetch_context_posts_expected_payload():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append({"path": request.url.path, "json": request.read().decode("utf-8")})
        return httpx.Response(200, json={"context": "Scoped memory"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver") as client:
        result = await prefetch_context(
            query="what were we doing",
            user_id="user-123",
            session_id="session-456",
            client=client,
        )

    assert result == "Scoped memory"
    assert payloads == [
        {
            "path": CONTEXT_PREFETCH_PATH,
            "json": '{"query":"what were we doing","user_id":"user-123","session_id":"session-456"}',
        }
    ]


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_backend_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == RAG_QUERY_PATH
        return httpx.Response(200, json={"answer": "The worker calls the RAG endpoint over HTTP."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver") as client:
        result = await search_knowledge_base(
            query="how does the voice worker reach the backend?",
            user_id="user-123",
            client=client,
        )

    assert result == "The worker calls the RAG endpoint over HTTP."
