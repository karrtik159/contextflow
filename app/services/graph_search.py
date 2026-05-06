"""
Neo4j graph database operations — entity and relationship queries.
"""

from __future__ import annotations

import threading

from neo4j import AsyncGraphDatabase

from app.core.config import settings

# ── Lazy, thread-safe Neo4j driver singleton ─────────────────
# The driver is created on first use instead of at import time,
# so importing this module no longer requires a running Neo4j.

_driver = None
_driver_lock = threading.Lock()


def _get_driver():
    """Return the shared async Neo4j driver, creating it on first call.

    Uses double-checked locking to avoid races when multiple threads
    (e.g. concurrent CrewAI tool calls) reach this simultaneously.
    """
    global _driver
    if _driver is not None:
        return _driver

    with _driver_lock:
        if _driver is not None:
            return _driver
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value()),
        )
    return _driver


async def get_driver():
    """Return the shared async Neo4j driver (lazy-initialized)."""
    return _get_driver()


async def find_related_entities(entity_name: str, max_hops: int = 2) -> list[dict]:
    """
    Traverse the knowledge graph to find entities related to `entity_name`
    within `max_hops` relationship hops.
    """
    # Neo4j requires a literal int for variable-length path patterns,
    # so we interpolate max_hops directly (already clamped to 1-3 by caller).
    safe_hops = max(1, min(int(max_hops), 3))
    query = f"""
    MATCH path = (start {{name: $name}})-[*1..{safe_hops}]-(end)
    RETURN
        start.name   AS entity,
        [r IN relationships(path) | type(r)] AS relationships,
        end.name     AS related_entity
    LIMIT 20
    """
    driver = _get_driver()
    async with driver.session() as session:
        result = await session.run(query, name=entity_name)
        return [record.data() async for record in result]


async def close_driver():
    """Gracefully close the Neo4j driver (call on app shutdown)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
