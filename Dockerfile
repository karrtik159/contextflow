# syntax=docker/dockerfile:1
# ============================================================
# Dockerfile — OpenAI Clone FastAPI Application
# Optimized multi-stage build for maximum speed & minimal size
# ============================================================

# ── Stage 1: Dependency Cache ────────────────────────────────
# This stage ONLY installs Python deps. It is cached as long as
# pyproject.toml doesn't change — making rebuilds near-instant
# when only application code changes.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS deps

WORKDIR /build

# System build deps (gcc for C extensions like asyncpg, bcrypt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest + lockfile (cache-busts only when deps change)
COPY pyproject.toml uv.lock ./

# Install deps with uv sync + Docker cache mount for blazing fast rebuilds
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync


# ── Stage 2: Source Preparation ──────────────────────────────
# Separate stage so source code changes don't re-trigger dep install.
FROM deps AS source

WORKDIR /build

# Copy application source (changes often → last layer)
COPY app/ ./app/
COPY agents/ ./agents/

# Copy Alembic config if present (graceful)
COPY alembic.ini* ./
COPY alembic/ ./alembic/


# ── Stage 3: Production Runtime ──────────────────────────────
# Minimal image — no gcc, no build tools, no pip cache.
FROM python:3.11-slim AS runtime

# Labels for image metadata
LABEL maintainer="karrtik159"
LABEL description="OpenAI Clone — Real-time Voice & Deep Memory AI"

WORKDIR /app

# Runtime-only system deps (libpq for asyncpg, curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built venv from deps stage (no compiler artifacts)
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source from source stage
COPY --from=source /build/app/ ./app/
COPY --from=source /build/agents/ ./agents/
COPY --from=source /build/alembic/ ./alembic/
COPY --from=source /build/alembic.ini* ./

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Uvicorn with production settings
CMD ["/opt/venv/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
