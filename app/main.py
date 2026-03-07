"""
FastAPI application entry point.

Lifespan events handle startup/shutdown for:
  - Database engine
  - Neo4j driver
  - Redis (future)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.router import api_router
from app.core.config import get_settings
from app.services.graph_search import close_driver as close_neo4j

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("🚀 Starting {name} ({env})", name=settings.app_name, env=settings.app_env)
    yield
    # Cleanup
    logger.info("🛑 Shutting down...")
    await close_neo4j()


app = FastAPI(
    title=settings.app_name,
    description="Real-time Voice & Deep Memory AI — Hybrid Graph-Vector RAG",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.app_debug,
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "environment": settings.app_env}
