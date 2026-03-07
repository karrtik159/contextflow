"""
Pydantic V2 schemas for Message — Create / Read.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageCreate(BaseModel):
    """Fields required to store a new message."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class MessageRead(BaseModel):
    """Response schema for a single message."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    token_count: int | None
    created_at: datetime
