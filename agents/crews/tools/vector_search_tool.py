"""
CrewAI Tool — Semantic Vector Search via pgvector.

Allows CrewAI agents to search for semantically similar messages
using cosine distance on OpenAI embeddings stored in PostgreSQL.
"""

import asyncio
from typing import Type

from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import settings


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
        # 1. Generate embedding for the query
        client = OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small",
        )
        query_embedding = response.data[0].embedding

        # 2. Run the async vector search synchronously (CrewAI tools are sync)
        from app.services.vector_search import search_similar_messages
        from app.core.db import async_session

        async def _search():
            async with async_session() as db:
                results = await search_similar_messages(
                    db=db,
                    query_embedding=query_embedding,
                    limit=limit,
                )
                return results

        results = asyncio.run(_search())

        if not results:
            return "No similar messages found in the vector database."

        # 3. Format results for the agent
        output_lines = [f"Found {len(results)} relevant results:\n"]
        for i, msg in enumerate(results, 1):
            output_lines.append(
                f"{i}. [{msg.role}] {msg.content[:300]}..."
                if len(msg.content) > 300
                else f"{i}. [{msg.role}] {msg.content}"
            )
        return "\n".join(output_lines)
