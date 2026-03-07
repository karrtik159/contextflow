"""
Shared FastAPI dependencies — DB session, current user, etc.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db

# Reusable type alias for injecting the DB session into route handlers.
DBSession = Annotated[AsyncSession, Depends(get_db)]
