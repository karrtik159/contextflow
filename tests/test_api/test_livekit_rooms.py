from jose import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_create_room_token_includes_chat_session_claims(monkeypatch):
    monkeypatch.setattr("app.api.v1.livekit_rooms.settings.LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setattr("app.api.v1.livekit_rooms.settings.LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setattr(
        "app.api.v1.livekit_rooms.settings.LIVEKIT_API_SECRET",
        type("Secret", (), {"get_secret_value": lambda self: "a" * 32})(),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/livekit/token",
            json={
                "user_id": "user-123",
                "room_name": "atlas-room",
                "chat_session_id": "session-456",
            },
        )

    assert response.status_code == 200
    data = response.json()
    claims = jwt.get_unverified_claims(data["token"])

    assert data["identity"] == "user-123"
    assert data["room"] == "atlas-room"
    assert claims["sub"] == "user-123"
    assert claims["attributes"] == {"chat_session_id": "session-456"}
    assert claims["metadata"] == '{"chat_session_id":"session-456"}'


@pytest.mark.asyncio
async def test_create_room_token_requires_livekit_configuration(monkeypatch):
    monkeypatch.setattr("app.api.v1.livekit_rooms.settings.LIVEKIT_URL", "")
    monkeypatch.setattr("app.api.v1.livekit_rooms.settings.LIVEKIT_API_KEY", "")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/livekit/token",
            json={"user_id": "user-123", "room_name": "atlas-room"},
        )

    assert response.status_code == 503
    assert "LiveKit is not configured" in response.json()["detail"]
