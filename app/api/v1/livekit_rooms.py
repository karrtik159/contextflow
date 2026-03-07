"""
LiveKit room & token management endpoints.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/livekit", tags=["LiveKit"])


@router.post("/token")
async def create_room_token(user_id: str, room_name: str):
    """
    Generate a LiveKit access token for a user to join a voice room.

    TODO: Implement with `livekit.api.AccessToken` once LiveKit SDK
    is verified against current docs.
    """
    return {
        "token": "PLACEHOLDER",
        "room": room_name,
        "identity": user_id,
        "note": "Implement with livekit.api.AccessToken — verify against docs.livekit.io",
    }
