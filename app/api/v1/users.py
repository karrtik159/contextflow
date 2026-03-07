"""
User CRUD endpoints.

FastCRUD API:
  - get(db, schema_to_select=..., return_as_model=True, **filters)
  - update(db, object=SchemaOrDict, **filters)
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.deps import DBSession
from app.schemas.user import UserRead, UserUpdate
from app.services.crud import user_crud

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID, db: DBSession):
    """Retrieve a single user by ID."""
    user = await user_crud.get(
        db,
        schema_to_select=UserRead,
        return_as_model=True,
        id=user_id,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(user_id: UUID, payload: UserUpdate, db: DBSession):
    """Partially update a user."""
    updated = await user_crud.update(
        db,
        object=payload,
        schema_to_select=UserRead,
        return_as_model=True,
        id=user_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated
