"""
Auth endpoints — register, login, token refresh.

FastCRUD.create() requires a Pydantic schema instance (calls .model_dump()),
so we use UserCreateInternal whose fields match the ORM columns exactly.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas.user import UserCreate, UserCreateInternal, UserRead
from app.services.crud import user_crud

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DBSession):
    """Register a new user."""
    existing = await user_crud.get(db, email=payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Build an internal schema that maps to ORM columns
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


@router.post("/login")
async def login(payload: UserCreate, db: DBSession):
    """Authenticate and return a JWT access token."""
    user = await user_crud.get(db, email=payload.email)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": str(user["id"])})
    return {"access_token": token, "token_type": "bearer"}
