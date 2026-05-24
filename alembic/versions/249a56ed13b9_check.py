"""Keep Mem0 storage SDK-managed.

Revision ID: 249a56ed13b9
Revises: 8ad35117ed9d
Create Date: 2026-03-09 02:58:42.099571

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "249a56ed13b9"
down_revision: str | Sequence[str] | None = "8ad35117ed9d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Do not manage Mem0 SDK-owned pgvector tables in Alembic."""


def downgrade() -> None:
    """Do not manage Mem0 SDK-owned pgvector tables in Alembic."""
