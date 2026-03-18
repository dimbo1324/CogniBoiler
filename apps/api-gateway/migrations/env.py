"""
Alembic migration environment for CogniBoiler API Gateway.

Configured for async SQLAlchemy (asyncpg driver).
Reads database URL from application settings — not from alembic.ini —
so the same .env file controls both the app and migrations.

Running migrations:
    # Apply all pending migrations:
    uv run --package api-gateway alembic -c apps/api-gateway/alembic.ini upgrade head

    # Generate a new migration after changing models:
    uv run --package api-gateway alembic -c apps/api-gateway/alembic.ini \
        revision --autogenerate -m "add_users_table"

    # Roll back one migration:
    uv run --package api-gateway alembic -c apps/api-gateway/alembic.ini downgrade -1
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context

# ─── Import application models ────────────────────────────────────────────────
# All models must be imported here so Alembic can detect schema changes.
# Adding a new model → import it here → alembic revision --autogenerate
# will pick it up automatically.
from api_gateway.config import settings
from api_gateway.models.user import Base  # noqa: F401 — registers all models
from sqlalchemy.ext.asyncio import create_async_engine

# ─── Alembic config ───────────────────────────────────────────────────────────

config = context.config

# Read logging config from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from application settings (reads .env)
config.set_main_option("sqlalchemy.url", settings.database_url)

# Metadata for --autogenerate support
target_metadata = Base.metadata


# ─── Offline migrations (no DB connection needed) ─────────────────────────────


def run_migrations_offline() -> None:
    """
    Run migrations in offline mode — generates SQL without connecting.

    Useful for reviewing what SQL will be executed before applying,
    or for generating SQL scripts to run manually on a remote DB.

    Usage:
        alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ─── Online migrations (connects to DB) ───────────────────────────────────────


def do_run_migrations(connection: object) -> None:
    """Configure and run migrations with an active DB connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Create an async engine and run migrations within a sync context.

    Alembic's migration functions are synchronous — we run them inside
    run_sync() which temporarily provides a synchronous connection
    from the async engine.
    """
    engine = create_async_engine(settings.database_url)

    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration — runs the async function."""
    asyncio.run(run_async_migrations())


# ─── Entry point ─────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
