"""
Core configuration — modular Pydantic BaseSettings grouped by concern.

Each sub-settings class handles one domain (app, auth, DB, Redis, etc.).
The final `Settings` class composes them all via multiple inheritance,
loading values from .env automatically.

Usage:
    from app.core.config import settings
    print(settings.POSTGRES_URI)
"""

import os
from enum import Enum

from pydantic import SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Application ─────────────────────────────────────────────
class AppSettings(BaseSettings):
    APP_NAME: str = "ContextFlow"
    APP_DESCRIPTION: str | None = "Real-time Voice & Deep Memory AI"
    APP_VERSION: str | None = "0.1.0"
    LICENSE_NAME: str | None = "MIT"
    CONTACT_NAME: str | None = None
    CONTACT_EMAIL: str | None = None


# ── Environment ─────────────────────────────────────────────
class EnvironmentOption(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(BaseSettings):
    ENVIRONMENT: EnvironmentOption = EnvironmentOption.LOCAL


# ── Auth / JWT ──────────────────────────────────────────────
class CryptSettings(BaseSettings):
    SECRET_KEY: SecretStr = SecretStr("super-secret-change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7


# ── PostgreSQL ──────────────────────────────────────────────
class PostgresSettings(BaseSettings):
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "contextflow_d"
    POSTGRES_ASYNC_PREFIX: str = "postgresql+asyncpg://"
    POSTGRES_SYNC_PREFIX: str = "postgresql://"
    POSTGRES_URL: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def POSTGRES_URI(self) -> str:
        """Builds user:pass@host:port/db fragment."""
        credentials = f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
        location = f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return f"{credentials}@{location}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Full async connection URL for SQLAlchemy / Alembic."""
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        return f"{self.POSTGRES_ASYNC_PREFIX}{self.POSTGRES_URI}"


# ── Neo4j ───────────────────────────────────────────────────
class Neo4jSettings(BaseSettings):
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr = SecretStr("changeme")


# ── Redis (Cache + Session) ─────────────────────────────────
class RedisCacheSettings(BaseSettings):
    REDIS_CACHE_HOST: str = "localhost"
    REDIS_CACHE_PORT: int = 6379

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_CACHE_URL(self) -> str:
        return f"redis://{self.REDIS_CACHE_HOST}:{self.REDIS_CACHE_PORT}"


# ── Redis Rate Limiter ──────────────────────────────────────
class RedisRateLimiterSettings(BaseSettings):
    REDIS_RATE_LIMIT_HOST: str = "localhost"
    REDIS_RATE_LIMIT_PORT: int = 6379

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_RATE_LIMIT_URL(self) -> str:
        return f"redis://{self.REDIS_RATE_LIMIT_HOST}:{self.REDIS_RATE_LIMIT_PORT}"


class DefaultRateLimitSettings(BaseSettings):
    DEFAULT_RATE_LIMIT_LIMIT: int = 10
    DEFAULT_RATE_LIMIT_PERIOD: int = 3600


# ── CORS ────────────────────────────────────────────────────
class CORSSettings(BaseSettings):
    CORS_ORIGINS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]


# ── First Admin User (Seeding) ──────────────────────────────
class FirstUserSettings(BaseSettings):
    ADMIN_NAME: str = "admin"
    ADMIN_EMAIL: str = "admin@admin.com"
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "!Ch4ng3Th1sP4ssW0rd!"


# ── AI Services ─────────────────────────────────────────────
class AISettings(BaseSettings):
    # LLM Provider: "openai" | "google" | "openrouter"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4.1-mini"

    # OpenAI / OpenRouter
    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: str = ""

    # Google Gemini
    GOOGLE_API_KEY: SecretStr = SecretStr("")

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    HUGGINGFACE_API_KEY: SecretStr = SecretStr("")

    @field_validator("LLM_PROVIDER", "EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v


class LiveKitSettings(BaseSettings):
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: SecretStr = SecretStr("")


class RAGServiceSettings(BaseSettings):
    RAG_SERVICE_TOKEN: SecretStr = SecretStr("")


# ── Observability ───────────────────────────────────────────
class ObservabilitySettings(BaseSettings):
    LANGSMITH_API_KEY: SecretStr = SecretStr("")
    LANGSMITH_PROJECT: str = "contextflow_dev"


# ── Logging ─────────────────────────────────────────────────
class LoggerSettings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT_JSON: bool = False


# ── Composed Settings ───────────────────────────────────────
class Settings(
    AppSettings,
    EnvironmentSettings,
    CryptSettings,
    PostgresSettings,
    Neo4jSettings,
    RedisCacheSettings,
    RedisRateLimiterSettings,
    DefaultRateLimitSettings,
    CORSSettings,
    FirstUserSettings,
    AISettings,
    LiveKitSettings,
    RAGServiceSettings,
    ObservabilitySettings,
    LoggerSettings,
):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Module-level singleton — import this everywhere
settings = Settings()
