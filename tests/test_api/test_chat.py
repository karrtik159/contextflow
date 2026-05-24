from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.v1.chat import create_message
from app.schemas.message import MessageCreate


@pytest.mark.asyncio
async def test_create_message_embeds_user_message(monkeypatch):
    captured = {}

    async def fake_embed_text_async_safe(content):
        captured["embedded_content"] = content
        return [0.1, 0.2, 0.3]

    async def fake_create(db, object, schema_to_select, return_as_model):
        captured["db"] = db
        captured["object"] = object
        captured["schema_to_select"] = schema_to_select
        captured["return_as_model"] = return_as_model
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.v1.chat.embed_text_async_safe", fake_embed_text_async_safe)
    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)

    session_id = uuid4()
    await create_message(
        session_id,
        MessageCreate(role="user", content="Remember that I prefer concise deployment checklists."),
        db=object(),
    )

    assert captured["embedded_content"] == "Remember that I prefer concise deployment checklists."
    assert captured["object"].session_id == session_id
    assert captured["object"].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_create_message_does_not_embed_system_message(monkeypatch):
    captured = {}

    async def fail_embed_text_async_safe(content):
        raise AssertionError("system messages should not be embedded")

    async def fake_create(db, object, schema_to_select, return_as_model):
        captured["object"] = object
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.v1.chat.embed_text_async_safe", fail_embed_text_async_safe)
    monkeypatch.setattr("app.api.v1.chat.message_crud.create", fake_create)

    await create_message(
        uuid4(),
        MessageCreate(role="system", content="Internal instruction."),
        db=object(),
    )

    assert captured["object"].embedding is None
