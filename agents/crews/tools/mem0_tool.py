"""
CrewAI Tool — Mem0 Memory Operations.

Allows CrewAI agents to store and retrieve long-term user memories
via the Mem0 service (backed by pgvector + Neo4j).
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MemorySearchInput(BaseModel):
    """Input schema for memory search."""

    query: str = Field(description="Search query to find relevant user memories.")
    user_id: str = Field(description="The user ID to search memories for.")
    limit: int = Field(default=5, description="Maximum number of memories to return.")


class MemoryStoreInput(BaseModel):
    """Input schema for storing a memory."""

    content: str = Field(description="The memory content to store (fact, preference, entity).")
    user_id: str = Field(description="The user ID to associate this memory with.")


class MemorySearchTool(BaseTool):
    name: str = "memory_search"
    description: str = (
        "Search the long-term memory store for a user's past preferences, "
        "facts, and conversation history. Returns the most relevant memories "
        "for personalization. Use this to understand user context before answering."
    )
    args_schema: type[BaseModel] = MemorySearchInput

    def _run(self, query: str, user_id: str, limit: int = 5) -> str:
        """Search Mem0 for relevant user memories."""
        from app.memory.mem0_service import Mem0Service

        try:
            results = Mem0Service.get_client().search(query, user_id=user_id, limit=limit)
        except Exception as e:
            return f"Memory search error: {e}"

        if not results:
            return f"No memories found for user '{user_id}' matching '{query}'."

        output_lines = [f"Found {len(results)} memories for user '{user_id}':\n"]
        for i, mem in enumerate(results, 1):
            text = mem.get("memory", mem.get("text", str(mem)))
            score = mem.get("score", "N/A")
            output_lines.append(f"{i}. [score={score}] {text}")

        return "\n".join(output_lines)


class MemoryStoreTool(BaseTool):
    name: str = "memory_store"
    description: str = (
        "Store a new fact, preference, or entity into the user's long-term memory. "
        "Use this after extracting important information from conversations "
        "that should be remembered across sessions."
    )
    args_schema: type[BaseModel] = MemoryStoreInput

    def _run(self, content: str, user_id: str) -> str:
        """Store a memory via Mem0."""
        from app.memory.mem0_service import Mem0Service, clean_memory_content

        clean_content = clean_memory_content(content)
        if clean_content is None:
            return "Skipped memory store: extracted content was too low-signal to persist."

        try:
            Mem0Service.add_memory(clean_content, user_id=user_id)
            return f"Memory stored successfully for user '{user_id}': {clean_content[:100]}"
        except Exception as e:
            return f"Memory store error: {e}"
