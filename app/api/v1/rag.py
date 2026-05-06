"""
RAG endpoint — Hybrid Graph-Vector retrieval-augmented generation
with semantic intent routing for low-latency simple chat.

Request flow:
  1. classify_intent() — fast LLM call (~200ms) to determine if RAG is needed.
  2a. Simple chat → stream_direct_chat() — direct LLM response (sub-second).
  2b. Knowledge query → SupportCrew kickoff — full CrewAI orchestration.
  3. Optionally queue MemoryCrew as a background task.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import DBSession, OptionalUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# ── Request / Response Schemas ───────────────────────────────
class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The user's question.")
    user_id: str | None = Field(default=None, description="Optional user ID for personalized context.")
    stream: bool = Field(default=False, description="If true, stream the response for simple chat.")


class RAGQueryResponse(BaseModel):
    answer: str
    query: str
    user_id: str | None = None
    routed_to: str = Field(description="'cache' for semantic hit, 'direct' for simple chat, 'crewai' for RAG queries.")


# ── Background Memory Processing ────────────────────────────
def _process_memory_background(query: str, answer: str, user_id: str):
    """Fire-and-forget: extract entities from the Q&A and persist to graph."""
    import time
    from agents.crews.memory_crew import MemoryCrew

    transcript = f"User: {query}\nAssistant: {answer}"
    query_snippet = query[:80]

    logger.info("MemoryCrew started — user=%s query='%s'", user_id, query_snippet)
    t0 = time.perf_counter()

    try:
        result = MemoryCrew().crew().kickoff(
            inputs={"transcript": transcript, "user_id": user_id}
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "MemoryCrew completed in %.1fs — user=%s query='%s' result_len=%d",
            elapsed, user_id, query_snippet, len(str(result)),
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "MemoryCrew FAILED after %.1fs — user=%s query='%s'",
            elapsed, user_id, query_snippet,
        )


# ── RAG Query Endpoint ───────────────────────────────────────
@router.post(
    "/query",
    status_code=status.HTTP_200_OK,
    summary="Ask a question — auto-routed between direct chat and full RAG",
)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    db: DBSession,
    user: OptionalUser = None,
):
    """
    Smart-routed query endpoint:

    0. **Sanitize & Pre-Filter** — Scrubs PII and simplifies structural mapping natively using regex.
    1. **Neo4j Semantic Cache** — Bypasses all processing securely grabbing sub-sec cache globally or natively isolated.
    2. **Intent Classification** — A fast LLM call (~200ms) determines if the
       query needs knowledge retrieval or is simple conversation.
    3. **Direct Chat Path** — Greetings, small talk, and opinions bypass
       CrewAI entirely for sub-second responses.
    4. **RAG Path** — Factual and knowledge queries go through the full
       SupportCrew (pgvector + Neo4j + Mem0) pipeline.
    5. **Memory** — Both paths optionally trigger background MemoryCrew and populate graphs.
    """
    from app.services.llm_provider import classify_intent, direct_chat, stream_direct_chat
    from app.services.embeddings import embed_text_async_safe
    from app.services.cache_sanitizer import sanitize_query
    from app.services.semantic_cache import get_cached_response, populate_semantic_cache

    # Determine user_id from auth or request body
    user_id = request.user_id
    if user and not user_id:
        user_id = str(user.get("id", "anonymous"))

    effective_user_id = user_id or "anonymous"

    # ── Step 0: Sanitize Inbound Query ──────────────────────
    sanitized = sanitize_query(request.query)

    # ── Step 1: Neo4j Zero-Latency Caching ──────────────────
    # If PII exists, ensure we only lookup cache belonging exclusively to this user!
    cache_lookup_scoped_id = effective_user_id if sanitized.requires_isolation else None

    # We must generate an embedding for vector lookup natively
    query_embedding = await embed_text_async_safe(sanitized.normalized_query)
    
    if query_embedding:
        cached_answer = await get_cached_response(
            normalized_query=sanitized.normalized_query, 
            embedding=query_embedding, 
            user_id=cache_lookup_scoped_id
        )
        if cached_answer:
            logger.info("Semantic Cache Hit: '%s' via scoped '%s' isolation", request.query[:80], cache_lookup_scoped_id or 'global')
            return RAGQueryResponse(
                answer=cached_answer,
                query=request.query,
                user_id=user_id,
                routed_to="cache",
            )

    # ── Step 2: Intent Classification ──────────────────────
    needs_rag = await classify_intent(request.query)

    # ── Cache Callback Helper ───────────────────────────────
    def _fire_background_callbacks(answer: str):
        if effective_user_id != "anonymous":
            background_tasks.add_task(
                _process_memory_background,
                query=request.query,
                answer=answer,
                user_id=effective_user_id,
            )
        
        # Add to Semantic cache graph. Use strict ownership isolation if PII matched natively!
        if query_embedding:
            background_tasks.add_task(
                populate_semantic_cache,
                normalized_query=sanitized.normalized_query,
                embedding=query_embedding,
                answer=answer,
                user_id=cache_lookup_scoped_id,
                session_id=None
            )

    # ── Step 3: Direct Chat (fast path) ────────────────────
    if not needs_rag:
        logger.info("Intent: simple chat — bypassing CrewAI for '%s'", request.query[:80])

        if request.stream:
            # Stream chunks to the client AND collect them so we can cache the full answer afterwards
            collected_chunks: list[str] = []

            async def _stream_generator():
                async for chunk in stream_direct_chat(request.query):
                    collected_chunks.append(chunk)
                    yield chunk
                # After streaming completes, fire cache population + memory
                full_answer = "".join(collected_chunks)
                _fire_background_callbacks(full_answer)

            return StreamingResponse(
                _stream_generator(),
                media_type="text/plain",
            )

        # Non-streaming response
        answer = await direct_chat(request.query)

        _fire_background_callbacks(answer)

        return RAGQueryResponse(
            answer=answer,
            query=request.query,
            user_id=user_id,
            routed_to="direct",
        )

    # ── Step 4: Full RAG (CrewAI path) ─────────────────────
    import asyncio
    import time

    logger.info("Intent: knowledge query — running SupportCrew for '%s'", request.query[:80])

    from agents.crews.support_crew import SupportCrew

    def _run_crew_sync():
        return (
            SupportCrew()
            .crew()
            .kickoff(
                inputs={
                    "query": request.query,
                    "user_id": effective_user_id,
                }
            )
        )

    t0 = time.perf_counter()
    try:
        # Offload blocking CrewAI execution to thread pool so the
        # FastAPI event loop remains responsive for other requests.
        result = await asyncio.to_thread(_run_crew_sync)
        elapsed = time.perf_counter() - t0
        answer = str(result).strip()

        logger.info(
            "SupportCrew completed in %.1fs — query='%s' answer_len=%d",
            elapsed, request.query[:80], len(answer),
        )

        # Validate crew output — if empty or suspiciously short, fall back to direct LLM
        if not answer or len(answer) < 5:
            logger.warning(
                "SupportCrew returned empty/invalid result, falling back to direct LLM"
            )
            answer = await direct_chat(request.query)
            routed = "direct_fallback"
        else:
            routed = "crewai"

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "SupportCrew FAILED after %.1fs — query='%s'", elapsed, request.query[:80]
        )
        # Graceful degradation: fall back to direct LLM instead of 500
        answer = await direct_chat(request.query)
        routed = "direct_fallback"

    _fire_background_callbacks(answer)

    return RAGQueryResponse(
        answer=answer,
        query=request.query,
        user_id=user_id,
        routed_to=routed,
    )
