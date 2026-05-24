import pytest
from fastapi import HTTPException

from app.api.v1.memory import (
    MemoryAddRequest,
    MemorySearchRequest,
    add_memory,
    get_all_memories,
    search_memories,
)


@pytest.mark.asyncio
async def test_add_memory_uses_authenticated_user_and_thread(monkeypatch):
    captured = {}

    def fake_add_memory(content, user_id, metadata=None):
        captured["content"] = content
        captured["user_id"] = user_id
        captured["metadata"] = metadata
        return {"id": "memory-1"}

    monkeypatch.setattr("app.memory.mem0_service.Mem0Service.add_memory", fake_add_memory)

    response = await add_memory(
        MemoryAddRequest(
            content="  Atlas prefers concise deployment checklists.  ",
            user_id="user-123",
            metadata={"source": "test"},
        ),
        user={"id": "user-123"},
    )

    assert response.status == "created"
    assert response.data == {"id": "memory-1"}
    assert captured == {
        "content": "Atlas prefers concise deployment checklists.",
        "user_id": "user-123",
        "metadata": {"source": "test"},
    }


@pytest.mark.asyncio
async def test_add_memory_rejects_spoofed_user_id():
    with pytest.raises(HTTPException) as exc:
        await add_memory(
            MemoryAddRequest(content="Atlas prefers concise checklists.", user_id="other-user"),
            user={"id": "user-123"},
        )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_add_memory_skips_low_signal_content(monkeypatch):
    def fail_add_memory(*args, **kwargs):
        raise AssertionError("low-signal memory should not reach Mem0")

    monkeypatch.setattr("app.memory.mem0_service.Mem0Service.add_memory", fail_add_memory)

    response = await add_memory(
        MemoryAddRequest(content="none"),
        user={"id": "user-123"},
    )

    assert response.status == "skipped"
    assert response.data == {"reason": "low_signal_memory"}


@pytest.mark.asyncio
async def test_search_memories_uses_authenticated_user(monkeypatch):
    captured = {}

    def fake_search_memories(query, user_id, limit=5):
        captured["query"] = query
        captured["user_id"] = user_id
        captured["limit"] = limit
        return [{"memory": "Atlas prefers concise deployment checklists."}]

    monkeypatch.setattr("app.memory.mem0_service.Mem0Service.search_memories", fake_search_memories)

    response = await search_memories(
        MemorySearchRequest(query=" deployment   checklist ", user_id="user-123", limit=3),
        user={"id": "user-123"},
    )

    assert response.status == "ok"
    assert response.data == [{"memory": "Atlas prefers concise deployment checklists."}]
    assert captured == {"query": "deployment checklist", "user_id": "user-123", "limit": 3}


@pytest.mark.asyncio
async def test_get_all_memories_rejects_spoofed_path_user_id():
    with pytest.raises(HTTPException) as exc:
        await get_all_memories("other-user", user={"id": "user-123"})

    assert exc.value.status_code == 403
