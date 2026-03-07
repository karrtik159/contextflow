"""
Memory endpoints — view, search, and manage long-term user memories via Mem0.

These endpoints expose the Mem0 service for direct memory operations,
useful for debugging, admin panels, and the Gradio demo.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/memories", tags=["Memory"])


# ── Schemas ──────────────────────────────────────────────────
class MemoryAddRequest(BaseModel):
    content: str = Field(min_length=1, description="The memory content to store.")
    user_id: str = Field(description="User ID to associate the memory with.")
    metadata: dict | None = Field(default=None, description="Optional metadata tags.")


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Search query.")
    user_id: str = Field(description="User ID whose memories to search.")
    limit: int = Field(default=5, ge=1, le=20, description="Max results.")


class MemoryResponse(BaseModel):
    status: str
    data: list | dict | None = None


# ── Endpoints ────────────────────────────────────────────────
@router.post("/add", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memory(request: MemoryAddRequest):
    """Store a new memory for a user."""
    from app.memory.mem0_service import Mem0Service

    try:
        result = await Mem0Service.add_memory(
            content=request.content,
            user_id=request.user_id,
            metadata=request.metadata,
        )
        return MemoryResponse(status="created", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {e}")


@router.post("/search", response_model=MemoryResponse)
async def search_memories(request: MemorySearchRequest):
    """Search memories for a user by semantic similarity."""
    from app.memory.mem0_service import Mem0Service

    try:
        results = await Mem0Service.search_memories(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit,
        )
        return MemoryResponse(status="ok", data=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {e}")


@router.get("/{user_id}", response_model=MemoryResponse)
async def get_all_memories(user_id: str):
    """Get all memories for a user."""
    from app.memory.mem0_service import Mem0Service

    try:
        memories = await Mem0Service.get_all_memories(user_id=user_id)
        return MemoryResponse(status="ok", data=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {e}")
