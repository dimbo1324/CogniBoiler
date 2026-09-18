"""
Database access for alert-manager.

The alarm tables are created by the Alembic chain that the `migrate` job applies; this
service never creates schema. It checks at start-up that the tables exist.
"""

from __future__ import annotations

import os

from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DEFAULT_DATABASE_URL: str = (
    "postgresql+asyncpg://cogniboiler:cogniboiler@localhost:5432/cogniboiler"
)
REQUIRED_TABLES: tuple[str, ...] = ("alarm_events", "alarm_transitions")


def database_url() -> str:
    """The URL from ALERT_MANAGER_DATABASE_URL, then DATABASE_URL."""
    return os.getenv(
        "ALERT_MANAGER_DATABASE_URL", os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    )


def create_engine(url: str | None = None) -> AsyncEngine:
    return create_async_engine(
        url or database_url(), pool_pre_ping=True, hide_parameters=True
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def missing_tables(engine: AsyncEngine) -> list[str]:
    """Alarm tables the migrations have not created yet."""

    def table_names(connection: Connection) -> list[str]:
        return inspect(connection).get_table_names()

    async with engine.connect() as connection:
        existing = set(await connection.run_sync(table_names))
    return [table for table in REQUIRED_TABLES if table not in existing]
