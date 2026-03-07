"""
Neo4j graph database operations — entity and relationship queries.
"""

from neo4j import AsyncGraphDatabase

from app.core.config import settings

_driver = AsyncGraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value()),
)


async def get_driver():
    """Return the shared async Neo4j driver."""
    return _driver


async def find_related_entities(entity_name: str, max_hops: int = 2) -> list[dict]:
    """
    Traverse the knowledge graph to find entities related to `entity_name`
    within `max_hops` relationship hops.
    """
    query = """
    MATCH path = (start {name: $name})-[*1..$hops]-(end)
    RETURN
        start.name   AS entity,
        [r IN relationships(path) | type(r)] AS relationships,
        end.name     AS related_entity
    LIMIT 20
    """
    async with _driver.session() as session:
        result = await session.run(query, name=entity_name, hops=max_hops)
        return [record.data() async for record in result]


async def close_driver():
    """Gracefully close the Neo4j driver (call on app shutdown)."""
    await _driver.close()
