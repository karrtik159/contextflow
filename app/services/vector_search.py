"""
pgvector-powered semantic search operations.
"""

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message


async def search_similar_messages(
    db: AsyncSession,
    query_embedding: list[float],
    session_id: UUID | None = None,
    limit: int = 5,
) -> list[Message]:
    """
    Find the most semantically similar messages using cosine distance.

    Args:
        db: Async database session.
        query_embedding: 1536-dim float vector from OpenAI embedding model.
        session_id: Optional filter to scope search within a single chat session.
        limit: Max results to return.

    Returns:
        List of Message ORM objects ordered by similarity (closest first).
    """
    stmt = (
        select(Message)
        .where(Message.embedding.is_not(None))
        .order_by(Message.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    if session_id:
        stmt = stmt.where(Message.session_id == session_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())
