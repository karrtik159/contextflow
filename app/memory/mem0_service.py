"""
Mem0 service — wraps the Mem0 client for injecting and retrieving
long-term user memories.  Acts as the bridge between CrewAI crews
and the underlying Neo4j / pgvector stores.
"""

from mem0 import Memory

from app.core.config import get_settings

settings = get_settings()


def _build_mem0_config() -> dict:
    """
    Build Mem0 configuration dict.
    Adapt this as Mem0's config API evolves.
    """
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": settings.openai_api_key.get_secret_value(),
                "model": "gpt-4o-mini",
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": settings.database_url.replace("+asyncpg", ""),
                "collection_name": "mem0_memories",
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": settings.neo4j_uri,
                "username": settings.neo4j_user,
                "password": settings.neo4j_password.get_secret_value(),
            },
        },
    }


class Mem0Service:
    """Singleton-style wrapper around the Mem0 Memory client."""

    _instance: Memory | None = None

    @classmethod
    def get_client(cls) -> Memory:
        if cls._instance is None:
            cls._instance = Memory.from_config(config_dict=_build_mem0_config())
        return cls._instance

    @classmethod
    async def add_memory(cls, content: str, user_id: str, metadata: dict | None = None):
        """Store a new memory for a given user."""
        client = cls.get_client()
        return client.add(content, user_id=user_id, metadata=metadata or {})

    @classmethod
    async def search_memories(cls, query: str, user_id: str, limit: int = 5):
        """Retrieve memories most relevant to the query."""
        client = cls.get_client()
        return client.search(query, user_id=user_id, limit=limit)

    @classmethod
    async def get_all_memories(cls, user_id: str):
        """Get the complete memory profile for a user."""
        client = cls.get_client()
        return client.get_all(user_id=user_id)
