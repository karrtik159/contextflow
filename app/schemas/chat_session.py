"""
Pydantic V2 schemas for ChatSession — Create / Update / Read.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    """Fields to start a new chat session."""

    title: str = Field(default="New Chat", max_length=255)


class SessionUpdate(BaseModel):
    """Partial update for a session."""

    title: str | None = Field(default=None, max_length=255)
    summary: str | None = None


class SessionRead(BaseModel):
    """Response schema for a chat session."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    summary: str | None
    created_at: datetime
    updated_at: datetime
