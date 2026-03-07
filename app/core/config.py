"""
Core configuration — loads all environment variables via Pydantic BaseSettings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, auto-loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "OpenAI_Clone"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://postgres:changeme@localhost:5432/openai_clone"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("changeme")

    # --- Auth / JWT ---
    secret_key: SecretStr = SecretStr("super-secret-change-me-in-production")
    access_token_expire_minutes: int = 30

    # --- OpenAI ---
    openai_api_key: SecretStr = SecretStr("")

    # --- LiveKit ---
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: SecretStr = SecretStr("")

    # --- Mem0 ---
    mem0_api_key: SecretStr = SecretStr("")

    # --- Observability ---
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "openai-clone"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    return Settings()
