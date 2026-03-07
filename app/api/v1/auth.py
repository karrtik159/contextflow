"""
Auth endpoints — register, login, token refresh.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSession
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas.user import UserCreate, UserRead
from app.services.crud import user_crud

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DBSession):
    """Register a new user."""
    # Check uniqueness
    existing = await user_crud.get(db=db, email=payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = await user_crud.create(
        db=db,
        object=payload.model_dump(exclude={"password"}) | {"hashed_password": hash_password(payload.password)},
    )
    return user


@router.post("/login")
async def login(payload: UserCreate, db: DBSession):
    """Authenticate and return a JWT access token."""
    user = await user_crud.get(db=db, email=payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}
