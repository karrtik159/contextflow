"""
Mem0 service — wraps the Mem0 client for injecting and retrieving
long-term user memories.  Acts as the bridge between CrewAI crews
and the underlying Neo4j / pgvector stores.
"""

import threading

from mem0 import Memory

from app.core.config import settings


def _build_mem0_llm_config() -> dict:
    config = {
        "api_key": settings.OPENAI_API_KEY.get_secret_value(),
        "model": settings.LLM_MODEL,
    }
    if settings.OPENAI_BASE_URL:
        config["openai_base_url"] = settings.OPENAI_BASE_URL
    return {
        "provider": "openai",
        "config": config,
    }


def _build_mem0_embedding_config() -> dict:
    if settings.EMBEDDING_PROVIDER == "huggingface":
        config = {
            "model": settings.EMBEDDING_MODEL,
        }
        api_key = settings.HUGGINGFACE_API_KEY.get_secret_value()
        if api_key:
            config["api_key"] = api_key
        return {
            "provider": "huggingface",
            "config": config,
        }

    config = {
        "api_key": settings.OPENAI_API_KEY.get_secret_value(),
        "model": settings.EMBEDDING_MODEL,
    }
    if settings.OPENAI_BASE_URL:
        config["openai_base_url"] = settings.OPENAI_BASE_URL
    return {
        "provider": "openai",
        "config": config,
    }


def _build_mem0_config() -> dict:
    """Build Mem0 configuration dict."""
    return {
        "llm": _build_mem0_llm_config(),
        "embedder": _build_mem0_embedding_config(),
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": f"{settings.POSTGRES_SYNC_PREFIX}{settings.POSTGRES_URI}",
                "collection_name": "mem0_memories",
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": settings.NEO4J_URI,
                "username": settings.NEO4J_USER,
                "password": settings.NEO4J_PASSWORD.get_secret_value(),
            },
        },
    }


class Mem0Service:
    """Thread-safe singleton wrapper around the Mem0 Memory client.

    Uses double-checked locking so concurrent CrewAI tool threads
    don't race to create multiple ``Memory`` instances.
    """

    _instance: Memory | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_client(cls) -> Memory:
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            # Another thread may have initialised while we waited
            if cls._instance is not None:
                return cls._instance
            cls._instance = Memory.from_config(config_dict=_build_mem0_config())
        return cls._instance

    # NOTE: The methods below are intentionally *synchronous*.
    # The Mem0 SDK's add/search/get_all are blocking calls.
    # Marking them ``async`` would silently block the event loop.
    # Callers needing async should wrap with ``asyncio.to_thread()``.

    @classmethod
    def add_memory(cls, content: str, user_id: str, metadata: dict | None = None):
        """Store a new memory for a given user."""
        client = cls.get_client()
        return client.add(content, user_id=user_id, metadata=metadata or {})

    @classmethod
    def search_memories(cls, query: str, user_id: str, limit: int = 5):
        """Retrieve memories most relevant to the query."""
        client = cls.get_client()
        return client.search(query, user_id=user_id, limit=limit)

    @classmethod
    def get_all_memories(cls, user_id: str):
        """Get the complete memory profile for a user."""
        client = cls.get_client()
        return client.get_all(user_id=user_id)
