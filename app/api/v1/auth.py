"""
Auth endpoints — register, login (OAuth2 form), refresh, logout, current user.

Login uses OAuth2PasswordRequestForm (standard form-data with username + password).
The refresh token is stored as an httpOnly cookie for security.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUser, DBSession
from app.core.config import settings
from app.core.security import (
    TokenType,
    authenticate_user,
    blacklist_token,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_token,
)
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserCreateInternal, UserRead
from app.services.crud import user_crud

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DBSession):
    """Register a new user."""
    existing = await user_crud.get(db, email=payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    internal = UserCreateInternal(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    new_user = await user_crud.create(
        db,
        object=internal,
        schema_to_select=UserRead,
        return_as_model=True,
    )
    return new_user


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DBSession,
):
    """Authenticate via OAuth2 form and return access + refresh tokens.

    The access token is returned in the response body.
    The refresh token is set as an httpOnly secure cookie.
    """
    user = await authenticate_user(
        username_or_email=form_data.username,
        password=form_data.password,
        db=db,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username, email, or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Set refresh token as httpOnly cookie
    max_age = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=max_age,
    )

    return Token(access_token=access_token, token_type="bearer")


@router.post("/refresh", response_model=Token)
async def refresh_access_token(request: Request, db: DBSession):
    """Issue a new access token using the refresh token cookie."""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    token_data = await verify_token(refresh_token, TokenType.REFRESH, db)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    new_access_token = create_access_token(data={"sub": token_data.username_or_email})
    return Token(access_token=new_access_token, token_type="bearer")


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: Request,
    response: Response,
    db: DBSession,
    current_user: CurrentUser,
):
    """Blacklist the current access + refresh tokens and clear the cookie."""
    # Blacklist the access token from the Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        access_token = auth_header.split(" ", 1)[1]
        await blacklist_token(access_token, db)

    # Blacklist the refresh token from the cookie
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        await blacklist_token(refresh_token, db)

    # Clear the refresh cookie
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")

    return {"detail": "Successfully logged out"}


@router.get("/me", response_model=UserRead)
async def get_me(current_user: CurrentUser):
    """Return the currently authenticated user's profile.

    Validates the Bearer token via ``get_current_user`` dependency
    and returns only read-safe fields.
    """
    return UserRead.model_validate(current_user)
