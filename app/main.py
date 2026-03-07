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
from app.core.config import settings
from app.services.graph_search import close_driver as close_neo4j


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("🚀 Starting {name} ({env})", name=settings.APP_NAME, env=settings.ENVIRONMENT.value)
    yield
    logger.info("🛑 Shutting down...")
    await close_neo4j()


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION or "Real-time Voice & Deep Memory AI",
    version=settings.APP_VERSION or "0.1.0",
    lifespan=lifespan,
    debug=(settings.ENVIRONMENT == "local"),
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT.value}
