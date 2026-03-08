"""
Auth / Token schemas — Token response, TokenData for JWT payloads, TokenBlacklist for revocation.
"""

from datetime import datetime

from pydantic import BaseModel


class Token(BaseModel):
    """Response schema for login / refresh endpoints."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Decoded JWT payload data."""

    username_or_email: str


class TokenBlacklistCreate(BaseModel):
    """Schema for inserting a revoked token into the blacklist."""

    token: str
    expires_at: datetime
