"""
User CRUD endpoints.
"""

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import DBSession
from app.schemas.user import UserRead, UserUpdate
from app.services.crud import user_crud

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID, db: DBSession):
    """Retrieve a single user by ID."""
    return await user_crud.get(db=db, id=user_id)


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(user_id: UUID, payload: UserUpdate, db: DBSession):
    """Partially update a user."""
    return await user_crud.update(db=db, object=payload.model_dump(exclude_unset=True), id=user_id)
