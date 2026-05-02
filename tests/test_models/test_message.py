from app.core.config import settings
from app.models.message import Message


def test_message_embedding_dimension_matches_settings():
    assert Message.__table__.c.embedding.type.dim == settings.EMBEDDING_DIMENSIONS
