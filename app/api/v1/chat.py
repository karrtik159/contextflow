"""
Chat endpoints — sessions & messages.
"""

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import DBSession
from app.schemas.chat_session import SessionCreate, SessionRead
from app.schemas.message import MessageCreate, MessageRead
from app.services.crud import message_crud, session_crud

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(user_id: UUID, payload: SessionCreate, db: DBSession):
    """Start a new chat session for a user."""
    return await session_crud.create(db=db, object=payload.model_dump() | {"user_id": user_id})


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(session_id: UUID, db: DBSession):
    """Retrieve all messages in a session."""
    result = await message_crud.get_multi(db=db, session_id=session_id)
    return result.get("data", [])


@router.post("/sessions/{session_id}/messages", response_model=MessageRead, status_code=201)
async def create_message(session_id: UUID, payload: MessageCreate, db: DBSession):
    """Add a new message to a session."""
    return await message_crud.create(db=db, object=payload.model_dump() | {"session_id": session_id})
