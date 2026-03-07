"""
RAG endpoint — Hybrid Graph-Vector retrieval-augmented generation.

Accepts a user query, runs the SupportCrew (CrewAI), and returns
the synthesized answer. Optionally triggers the MemoryCrew as a
background task to process the conversation for long-term memory.
"""

from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser

router = APIRouter(prefix="/rag", tags=["RAG"])


# ── Request / Response Schemas ───────────────────────────────
class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The user's question.")
    user_id: str | None = Field(default=None, description="Optional user ID for personalized context.")


class RAGQueryResponse(BaseModel):
    answer: str
    query: str
    user_id: str | None = None


# ── Background Memory Processing ────────────────────────────
def _process_memory_background(query: str, answer: str, user_id: str):
    """Fire-and-forget: extract entities from the Q&A and persist to graph."""
    from agents.crews.memory_crew import MemoryCrew

    transcript = f"User: {query}\nAssistant: {answer}"
    try:
        MemoryCrew().crew().kickoff(inputs={"transcript": transcript, "user_id": user_id})
    except Exception:
        # Background task — log but don't propagate
        import logging

        logging.getLogger(__name__).exception("MemoryCrew background task failed")


# ── RAG Query Endpoint ───────────────────────────────────────
@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question using Hybrid Graph-Vector RAG",
)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    user: OptionalUser = None,
):
    """
    1. Runs the SupportCrew (CrewAI) to retrieve context from
       pgvector + Neo4j + Mem0 and synthesize an answer.
    2. Optionally queues a background MemoryCrew task to extract
       entities from the conversation for long-term memory.
    """
    from agents.crews.support_crew import SupportCrew

    # Determine user_id from auth or request body
    user_id = request.user_id
    if user and not user_id:
        user_id = str(user.get("id", "anonymous"))

    # Run the SupportCrew synchronously (CrewAI crews are sync)
    result = (
        SupportCrew()
        .crew()
        .kickoff(
            inputs={
                "query": request.query,
                "user_id": user_id or "anonymous",
            }
        )
    )

    answer = str(result)

    # Queue background memory processing if we have a user
    if user_id and user_id != "anonymous":
        background_tasks.add_task(
            _process_memory_background,
            query=request.query,
            answer=answer,
            user_id=user_id,
        )

    return RAGQueryResponse(
        answer=answer,
        query=request.query,
        user_id=user_id,
    )
