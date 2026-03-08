"""
CrewAI Tool — Semantic Vector Search via pgvector.

Allows CrewAI agents to search for semantically similar messages
using cosine distance on embeddings stored in PostgreSQL.

NOTE: CrewAI tools run synchronously and use asyncio.run() to call
async code. Each asyncio.run() creates a new event loop, so we MUST
create a fresh async engine per call — the module-level engine is
bound to the FastAPI event loop and will error here.
"""

import asyncio
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.services.embeddings import embed_text


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

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from app.core.config import settings
        from app.services.vector_search import search_similar_messages

        async def _search():
            # Create a fresh engine bound to THIS event loop
            engine = create_async_engine(settings.database_url, pool_pre_ping=True)
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            try:
                async with session_factory() as db:
                    return await search_similar_messages(
                        db=db,
                        query_embedding=query_embedding,
                        limit=limit,
                    )
            finally:
                await engine.dispose()

        try:
            results = asyncio.run(_search())
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
