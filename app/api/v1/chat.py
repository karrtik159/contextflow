"""
Chat endpoints — sessions & messages.

Uses *Internal schemas for FastCRUD create() which requires .model_dump().
"""

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import DBSession
from app.schemas.chat_session import SessionCreate, SessionCreateInternal, SessionRead
from app.schemas.message import MessageCreate, MessageCreateInternal, MessageRead
from app.services.crud import message_crud, session_crud

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(user_id: UUID, payload: SessionCreate, db: DBSession):
    """Start a new chat session for a user."""
    internal = SessionCreateInternal(
        title=payload.title,
        user_id=user_id,
    )
    new_session = await session_crud.create(
        db,
        object=internal,
        schema_to_select=SessionRead,
        return_as_model=True,
    )
    return new_session


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(session_id: UUID, db: DBSession):
    """Retrieve all messages in a session."""
    result = await message_crud.get_multi(
        db,
        schema_to_select=MessageRead,
        return_as_model=True,
        session_id=session_id,
    )
    # get_multi returns {"data": [...], "total_count": N}
    return result["data"]


@router.post("/sessions/{session_id}/messages", response_model=MessageRead, status_code=201)
async def create_message(session_id: UUID, payload: MessageCreate, db: DBSession):
    """Add a new message to a session."""
    internal = MessageCreateInternal(
        role=payload.role,
        content=payload.content,
        session_id=session_id,
    )
    new_msg = await message_crud.create(
        db,
        object=internal,
        schema_to_select=MessageRead,
        return_as_model=True,
    )
    return new_msg
