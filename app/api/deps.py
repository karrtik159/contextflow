"""
Shared FastAPI dependencies — DB session, current user, rate limiting.

All dependencies are designed as reusable `Annotated` type aliases:
    DBSession       → async database session (auto commit/rollback)
    CurrentUser     → authenticated user dict (requires valid JWT)
    OptionalUser    → user dict or None (no auth required)
    CurrentSuperUser → authenticated admin (requires is_superuser=True)
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.services.crud import user_crud

# ── OAuth2 scheme — extracts Bearer token from Authorization header ──
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")


# ── Database Session ─────────────────────────────────────────────────
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ── Current Authenticated User ───────────────────────────────────────
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DBSession,
) -> dict[str, Any]:
    """
    Verify JWT (access type, not blacklisted), look up the user,
    and return their data as a dict.
    Raises 401 if the token is invalid, blacklisted, or the user doesn't exist.
    """
    from app.core.security import TokenType, verify_token

    token_data = await verify_token(token, TokenType.ACCESS, db)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # JWT sub claim stores the username
    user = await user_crud.get(db, username=token_data.username_or_email)
    if not user:
        # Fallback: try by email (in case sub was set to email)
        user = await user_crud.get(db, email=token_data.username_or_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


# ── Optional User (no auth required) ────────────────────────────────
async def get_optional_user(
    request: Request,
    db: DBSession,
) -> dict[str, Any] | None:
    """
    Attempt to extract and validate a Bearer token from the request.
    Returns user dict if valid, None otherwise. Never raises.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    try:
        return await get_current_user(parts[1], db)
    except HTTPException:
        return None


OptionalUser = Annotated[dict[str, Any] | None, Depends(get_optional_user)]


# ── Superuser Gate ───────────────────────────────────────────────────
async def get_current_superuser(
    current_user: CurrentUser,
) -> dict[str, Any]:
    """
    Requires the authenticated user to have is_superuser=True.
    Raises 403 Forbidden otherwise.
    """
    if not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges",
        )
    return current_user


CurrentSuperUser = Annotated[dict[str, Any], Depends(get_current_superuser)]


# ── Rate Limiter (basic in-memory, upgrade to Redis later) ───────────
async def rate_limiter_dependency(
    request: Request,
    db: DBSession,
    user: OptionalUser = None,
) -> None:
    """
    Placeholder rate limiter dependency.

    TODO: Implement with Redis-backed sliding window counter.
    Config: settings.DEFAULT_RATE_LIMIT_LIMIT / settings.DEFAULT_RATE_LIMIT_PERIOD
    """
    # Will be implemented when Redis rate limiting module is built.
    # For now, all requests pass through.
    pass
