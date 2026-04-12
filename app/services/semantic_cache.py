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
                    logger.warning(f"Vector dimension mismatch! Found {current_dim}, expected {target_dim}. Rebuilding index...")
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


async def get_cached_response(normalized_query: str, embedding: list[float], user_id: str | None = None) -> str | None:
    """
    Performs a vector search in Neo4j to find a semantically identical prior question.
    If the query involves PII (thus providing a user_id), it isolates the vector
    match to that user. Otherwise, it searches the global cache.
    """
    driver = await get_driver()
    
    # Dynamically construct pre-filter conditions depending on isolation requirements
    if user_id:
        # PII detected: only return hits belonging to this specific user
        filter_clause = "WHERE c.user_id = $userId"
        params = {"embedding": embedding, "userId": user_id}
    else:
        # No PII: search only the global cache pool
        filter_clause = "WHERE c.user_id = 'global'"
        params = {"embedding": embedding}
        
    query = f"""
    CALL db.index.vector.queryNodes('semantic_cache_index', 3, $embedding)
    YIELD node AS c, score
    {filter_clause} AND score > 0.95
    RETURN c.answer AS answer
    ORDER BY score DESC LIMIT 1
    """

    async with driver.session() as session:
        result = await session.run(query, **params)
        record = await result.single()
        if record:
            return record["answer"]
        return None


async def populate_semantic_cache(
    normalized_query: str, 
    embedding: list[float], 
    answer: str, 
    user_id: str | None = None,
    session_id: str | None = None
):
    """
    Appends a finalized LLM response natively to the semantic cache inside Neo4j.
    If 'user_id' is supplied, this cached node stays strictly isolated out of global hits.
    """
    driver = await get_driver()

    query = """
    CREATE (c:SemanticCache {
        normalized_query: $normalized_query,
        embedding: $embedding,
        answer: $answer,
        user_id: $user_id,
        session_id: $session_id,
        timestamp: timestamp()
    })
    """
    
    params = {
        "normalized_query": normalized_query,
        "embedding": embedding,
        "answer": answer,
        "user_id": user_id if user_id else "global",
        "session_id": session_id
    }
    
    async with driver.session() as session:
        await session.run(query, **params)

