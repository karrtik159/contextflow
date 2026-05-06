"""
CrewAI Tool — Semantic Vector Search via pgvector.

Allows CrewAI agents to search for semantically similar messages
using cosine distance on embeddings stored in PostgreSQL.

Uses the shared ``run_async()`` bridge for event-loop safety and a
thread-local async engine to avoid per-invocation connection overhead.
"""

import threading
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from agents.crews.tools.async_bridge import run_async
from app.services.embeddings import embed_text

# ── Thread-local SQLAlchemy engine ───────────────────────────
# CrewAI tools run in threads that are separate from the FastAPI event
# loop.  We keep one async engine *per thread* so successive tool calls
# reuse the same connection pool rather than paying TCP + auth overhead
# on every invocation.

_thread_engines = threading.local()


def _get_thread_engine():
    """Return (or create) the async engine for the current thread."""
    engine = getattr(_thread_engines, "engine", None)
    if engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine

        from app.core.config import settings

        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=2,  # Minimal — each thread runs one query at a time
        )
        _thread_engines.engine = engine
    return engine


class VectorSearchInput(BaseModel):
    """Input schema for the vector search tool."""

    query: str = Field(description="The search query to find semantically similar content.")
    limit: int = Field(default=5, description="Maximum number of results to return.")


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "Search the vector database (PostgreSQL + pgvector) for messages "
        "semantically similar to the query. Returns ranked results with "
        "content and similarity context. Use this for finding relevant "
        "past conversations and documents."
    )
    args_schema: Type[BaseModel] = VectorSearchInput

    def _run(self, query: str, limit: int = 5) -> str:
        """Embed the query and search pgvector for similar messages."""
        query_embedding = embed_text(query)

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from app.services.vector_search import search_similar_messages

        async def _search():
            engine = _get_thread_engine()
            factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
            async with factory() as db:
                return await search_similar_messages(
                    db=db,
                    query_embedding=query_embedding,
                    limit=limit,
                )

        try:
            results = run_async(_search())
        except Exception as e:
            return f"Vector search error: {e}"

        if not results:
            return "No similar messages found in the vector database."

        output_lines = [f"Found {len(results)} relevant results:\n"]
        for i, msg in enumerate(results, 1):
            output_lines.append(
                f"{i}. [{msg.role}] {msg.content[:300]}..."
                if len(msg.content) > 300
                else f"{i}. [{msg.role}] {msg.content}"
            )
        return "\n".join(output_lines)

