"""
Alembic env.py — configured for async SQLAlchemy (asyncpg) with our project structure.

Key wiring:
  - Config:   app.core.config.get_settings()
  - Base:     app.core.db.Base
  - Models:   app.models (auto-imported via __init__.py)
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.core.config import get_settings
from app.core.db import Base

# Import all models so Base.metadata is fully populated
import app.models  # noqa: F401  — triggers User, ChatSession, Message imports

# ── Alembic Config ──────────────────────────────────────────
config = context.config
settings = get_settings()

# Override sqlalchemy.url with our async DATABASE_URL
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


# ── Offline Mode (generates SQL without DB connection) ──────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online Mode (async — required for asyncpg driver) ───────
def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ── Entrypoint ──────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
