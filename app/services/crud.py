"""
FastCRUD instances — one per model.

Schemas are passed at the route/call level, not in the constructor.
Usage in routers:
    from app.services.crud import user_crud
    users = await user_crud.get_multi(db=session, schema_to_select=UserRead)
"""

from fastcrud import FastCRUD

from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.message import Message

# FastCRUD only takes the SQLAlchemy model — schemas are used per-operation
user_crud = FastCRUD(User)
session_crud = FastCRUD(ChatSession)
message_crud = FastCRUD(Message)
