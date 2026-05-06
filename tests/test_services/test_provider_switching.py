from types import SimpleNamespace

from app.services.embeddings import _validate_embedding_dimensions, embed_text
from app.services.llm_provider import build_crewai_llm


def test_build_crewai_llm_uses_configured_model(monkeypatch):
    monkeypatch.setattr("app.services.llm_provider.settings.LLM_PROVIDER", "openrouter")
    monkeypatch.setattr("app.services.llm_provider.settings.LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setattr("app.services.llm_provider.settings.OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        "app.services.llm_provider.settings.OPENAI_API_KEY",
        type("Secret", (), {"get_secret_value": lambda self: "test-key"})(),
    )

    llm = build_crewai_llm()

    assert llm.model.endswith("gpt-4o-mini")
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.api_key == "test-key"


def test_validate_embedding_dimensions_rejects_mismatch(monkeypatch):
    monkeypatch.setattr("app.services.embeddings.settings.EMBEDDING_DIMENSIONS", 1536)

    try:
        _validate_embedding_dimensions([0.1, 0.2, 0.3])
    except ValueError as exc:
        assert "Embedding dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected a dimension mismatch error")


def test_embed_text_huggingface_uses_sentence_transformer(monkeypatch):
    monkeypatch.setattr("app.services.embeddings.settings.EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr("app.services.embeddings.settings.EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setattr("app.services.embeddings.settings.EMBEDDING_DIMENSIONS", 3)

    class FakeSentenceTransformer:
        def encode(self, text, *, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
            assert text == "test query"
            return SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])

    monkeypatch.setattr("app.services.embeddings._get_hf_encoder", lambda: FakeSentenceTransformer())

    assert embed_text("test query") == [0.1, 0.2, 0.3]
