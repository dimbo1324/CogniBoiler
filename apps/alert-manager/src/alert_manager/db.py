"""Database helpers for alert-manager."""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alert_manager.models import Base

DATABASE_URL: str = os.getenv(
    "ALERT_MANAGER_DATABASE_URL",
    os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://cogniboiler:cogniboiler@localhost:5432/cogniboiler",
    ),
)

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def init_db() -> None:
    """Create the alarm table if it does not yet exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Create a short-lived async session for ad-hoc operations."""
    return AsyncSessionLocal()
