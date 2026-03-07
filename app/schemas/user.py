"""
Pydantic V2 schemas for User — Create / Update / Read.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    """Shared fields across all user schemas."""

    email: EmailStr
    username: str = Field(min_length=3, max_length=100)


class UserCreate(UserBase):
    """API-facing schema — used for request validation."""

    password: str = Field(min_length=8, max_length=128)


class UserCreateInternal(UserBase):
    """Internal schema — maps directly to ORM columns for FastCRUD create()."""

    hashed_password: str


class UserUpdate(BaseModel):
    """Partial update — all fields optional."""

    email: EmailStr | None = None
    username: str | None = Field(default=None, min_length=3, max_length=100)
    password: str | None = Field(default=None, min_length=8, max_length=128)
    is_active: bool | None = None


class UserRead(UserBase):
    """Response schema — never exposes password."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
