import logging

from app.services.graph_search import get_driver

logger = logging.getLogger(__name__)


async def init_semantic_cache():
    """
    Ensures that the necessary Vector Index exists in Neo4j for fast semantic lookups.
    Dynamically rebuilds the index if the underlying Embedder Dimension (e.g. OpenAI->HuggingFace) shifts.
    """
    from app.core.config import settings
    driver = await get_driver()
    target_dim = settings.EMBEDDING_DIMENSIONS
    
    async with driver.session() as session:
        # Check if index exists and what dimension it holds
        result = await session.run("SHOW VECTOR INDEXES YIELD name, options WHERE name = 'semantic_cache_index'")
        record = await result.single()
        should_create = True
        
        if record:
            try:
                options = record.get("options", {})
                index_config = options.get("indexConfig", {})
                current_dim = index_config.get("vector.dimensions")
                
                if current_dim and int(current_dim) != int(target_dim):
                    logger.warning(
                        "Vector dimension mismatch! Found %s, expected %s. Rebuilding index...",
                        current_dim,
                        target_dim,
                    )
                    await session.run("DROP INDEX semantic_cache_index")
                else:
                    should_create = False
            except Exception as e:
                logger.error(f"Error checking index dimensions: {e}")
                
        if should_create:
            query_create = f"""
            CREATE VECTOR INDEX semantic_cache_index IF NOT EXISTS
            FOR (c:SemanticCache) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {target_dim},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
            await session.run(query_create)
            logger.info("SemanticCache vector index initialized.")


async def get_cached_response(
    normalized_query: str,
    embedding: list[float],
    user_id: str,
    *,
    candidate_count: int = 50,
    score_threshold: float = 0.95,
) -> str | None:
    """
    Return a cached answer for this user only.

    Exact normalized-query matches win. If no exact match exists, use the vector
    index with a larger candidate set, then filter to the user's cache entries.
    """
    driver = await get_driver()

    exact_query = """
    MATCH (c:SemanticCache {user_id: $user_id, normalized_query: $normalized_query})
    RETURN c.answer AS answer
    ORDER BY c.timestamp DESC
    LIMIT 1
    """

    vector_query = """
    CALL db.index.vector.queryNodes('semantic_cache_index', $candidate_count, $embedding)
    YIELD node AS c, score
    WHERE c.user_id = $user_id AND score > $score_threshold
    RETURN c.answer AS answer
    ORDER BY score DESC LIMIT 1
    """

    async with driver.session() as session:
        result = await session.run(
            exact_query,
            user_id=user_id,
            normalized_query=normalized_query,
        )
        record = await result.single()
        if record:
            return record["answer"]

        result = await session.run(
            vector_query,
            embedding=embedding,
            user_id=user_id,
            candidate_count=max(1, candidate_count),
            score_threshold=score_threshold,
        )
        record = await result.single()
        if record:
            return record["answer"]
        return None


async def populate_semantic_cache(
    normalized_query: str, 
    embedding: list[float], 
    answer: str, 
    user_id: str,
    session_id: str | None = None
):
    """
    Upsert a finalized RAG response in the user's semantic cache.
    """
    driver = await get_driver()

    query = """
    MERGE (c:SemanticCache {
        user_id: $user_id,
        normalized_query: $normalized_query
    })
    SET
        c.embedding = $embedding,
        c.answer = $answer,
        c.session_id = $session_id,
        c.timestamp = timestamp()
    """
    
    params = {
        "normalized_query": normalized_query,
        "embedding": embedding,
        "answer": answer,
        "user_id": user_id,
        "session_id": session_id
    }
    
    async with driver.session() as session:
        await session.run(query, **params)
