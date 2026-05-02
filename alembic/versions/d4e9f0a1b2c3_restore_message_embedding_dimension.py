"""restore message embedding dimension

Revision ID: d4e9f0a1b2c3
Revises: 249a56ed13b9
Create Date: 2026-05-02 20:45:00.000000

"""
from collections.abc import Sequence

import pgvector.sqlalchemy

from alembic import op
from app.core.config import settings

# revision identifiers, used by Alembic.
revision: str = "d4e9f0a1b2c3"
down_revision: str | Sequence[str] | None = "249a56ed13b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TARGET_DIMENSIONS = settings.EMBEDDING_DIMENSIONS


def upgrade() -> None:
    """Upgrade schema."""
    if TARGET_DIMENSIONS == 384:
        return

    op.drop_index(
        "ix_messages_embedding_hnsw",
        table_name="messages",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.execute("UPDATE messages SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "messages",
        "embedding",
        existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=384),
        type_=pgvector.sqlalchemy.vector.VECTOR(dim=TARGET_DIMENSIONS),
        existing_nullable=True,
    )
    op.create_index(
        "ix_messages_embedding_hnsw",
        "messages",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    if TARGET_DIMENSIONS == 384:
        return

    op.drop_index(
        "ix_messages_embedding_hnsw",
        table_name="messages",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.execute("UPDATE messages SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column(
        "messages",
        "embedding",
        existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=TARGET_DIMENSIONS),
        type_=pgvector.sqlalchemy.vector.VECTOR(dim=384),
        existing_nullable=True,
    )
    op.create_index(
        "ix_messages_embedding_hnsw",
        "messages",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
