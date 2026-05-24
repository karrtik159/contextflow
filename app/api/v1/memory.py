"""
Authenticated memory endpoints for user-scoped Mem0 operations.

These endpoints expose the self-hosted Mem0 service for debugging, admin
surfaces, and the Gradio demo while enforcing the authenticated user's scope.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, get_user_id_value
from app.memory.mem0_service import clean_memory_content

router = APIRouter(prefix="/memories", tags=["Memory"])


class MemoryAddRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000, description="The memory content to store.")
    user_id: str | None = Field(
        default=None,
        description="Optional legacy user id; when supplied it must match the authenticated user.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata tags.")


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Search query.")
    user_id: str | None = Field(
        default=None,
        description="Optional legacy user id; when supplied it must match the authenticated user.",
    )
    limit: int = Field(default=5, ge=1, le=20, description="Max results.")


class MemoryResponse(BaseModel):
    status: str
    data: list | dict | None = None


def _resolve_memory_user_id(request_user_id: str | None, user: dict[str, Any]) -> str:
    auth_user_id = get_user_id_value(user)
    if auth_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user is missing an id",
        )

    if request_user_id and request_user_id != auth_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="request user_id does not match authenticated user",
        )
    return auth_user_id


@router.post("/add", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memory(request: MemoryAddRequest, user: CurrentUser):
    """Store a new memory for the authenticated user."""
    from app.memory.mem0_service import Mem0Service

    user_id = _resolve_memory_user_id(request.user_id, user)
    clean_content = clean_memory_content(request.content)
    if clean_content is None:
        return MemoryResponse(status="skipped", data={"reason": "low_signal_memory"})

    try:
        result = await asyncio.to_thread(
            Mem0Service.add_memory,
            clean_content,
            user_id,
            request.metadata,
        )
        return MemoryResponse(status="created", data=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {exc}") from exc


@router.post("/search", response_model=MemoryResponse)
async def search_memories(request: MemorySearchRequest, user: CurrentUser):
    """Search memories for the authenticated user by semantic similarity."""
    from app.memory.mem0_service import Mem0Service

    user_id = _resolve_memory_user_id(request.user_id, user)
    query = " ".join(request.query.strip().split())
    if not query:
        return MemoryResponse(status="ok", data=[])

    try:
        results = await asyncio.to_thread(
            Mem0Service.search_memories,
            query,
            user_id,
            request.limit,
        )
        return MemoryResponse(status="ok", data=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {exc}") from exc


@router.get("/{user_id}", response_model=MemoryResponse)
async def get_all_memories(user_id: str, user: CurrentUser):
    """Get all memories for the authenticated user."""
    from app.memory.mem0_service import Mem0Service

    resolved_user_id = _resolve_memory_user_id(user_id, user)
    try:
        memories = await asyncio.to_thread(Mem0Service.get_all_memories, resolved_user_id)
        return MemoryResponse(status="ok", data=memories)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {exc}") from exc
