from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.embeddings import embed_text_async
from app.services.vector_search import search_similar_messages

router = APIRouter(prefix="/context", tags=["Context"])


class PrefetchRequest(BaseModel):
    query: str
    user_id: UUID | str
    session_id: UUID | str | None = None
    limit: int = 5


class PrefetchResponse(BaseModel):
    context: str


def _parse_uuid(value: UUID | str | None) -> UUID | None:
    if value in (None, "", "anonymous"):
        return None

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


@router.post("/prefetch", response_model=PrefetchResponse)
async def prefetch_context(
    request: PrefetchRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Ultra-low latency semantic search for Voice AI RAG.

    This endpoint explicitly bypasses the CrewAI orchestration layer.
    It takes the STT transcript the exact millisecond the user stops speaking,
    performs a raw pgvector HNSW search, and formats the context for direct
    LLM injection via LiveKit's `on_user_turn_completed`.
    """
    user_uuid = _parse_uuid(request.user_id)
    session_uuid = _parse_uuid(request.session_id)
    if user_uuid is None and session_uuid is None:
        return {"context": "No scoped conversation context is available yet."}

    try:
        query_embedding = await embed_text_async(request.query)

        similar_messages = await search_similar_messages(
            db=db,
            query_embedding=query_embedding,
            session_id=session_uuid,
            user_id=user_uuid,
            limit=request.limit,
        )

        if not similar_messages:
            return {"context": "No highly relevant past context found."}

        context_lines = []
        for msg in similar_messages:
            context_lines.append(f"[{msg.role.upper()}]: {msg.content}")

        context_block = "\n".join(context_lines)
        return {"context": f"Relevant context from previous memory/conversations:\n{context_block}"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context prefetch failed: {str(e)}",
        )
