"""
LiveKit room & token management endpoints.

Generates LiveKit access tokens for users to join voice rooms.

LiveKit Agents SDK 1.4+ — verified against docs.livekit.io (March 2026).
"""

import json

from fastapi import APIRouter, HTTPException
from livekit.api import AccessToken, VideoGrants
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(prefix="/livekit", tags=["LiveKit"])


class TokenRequest(BaseModel):
    user_id: str = Field(description="Identity of the user joining the room.")
    room_name: str = Field(description="Name of the LiveKit room to join.")
    chat_session_id: str | None = Field(
        default=None,
        description="Optional chat session ID passed through to the voice worker for scoped RAG prefetch.",
    )


class TokenResponse(BaseModel):
    token: str
    room: str
    identity: str


@router.post("/token", response_model=TokenResponse)
async def create_room_token(request: TokenRequest):
    """
    Generate a LiveKit access token for a user to join a voice room.

    The token grants permission to join the specified room with
    audio publish/subscribe capabilities.
    """
    if not settings.LIVEKIT_URL or not settings.LIVEKIT_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET.",
        )

    token = (
        AccessToken(
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET.get_secret_value(),
        )
        .with_identity(request.user_id)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=request.room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    if request.chat_session_id:
        token = token.with_attributes({"chat_session_id": request.chat_session_id}).with_metadata(
            json.dumps({"chat_session_id": request.chat_session_id}, separators=(",", ":"))
        )

    return TokenResponse(
        token=token.to_jwt(),
        room=request.room_name,
        identity=request.user_id,
    )
