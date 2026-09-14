"""
FastAPI dependency injection for database sessions.

Provides get_db() — an async SQLAlchemy session factory used by all
routers that need database access.

Usage in a router:
    from api_gateway.dependencies import DbSession

    @router.get("/users")
    async def list_users(db: DbSession) -> list[UserResponse]:
        result = await db.execute(select(User))
        return result.scalars().all()

Connection pool:
    AsyncEngine uses a connection pool (default: 5 connections, max 10).
    Sessions are acquired per-request and released at the end via the
    async generator — FastAPI calls the cleanup code after the response
    is sent.

Configuration:
    database_url is read from settings (env var DATABASE_URL or .env file).
    Format: postgresql+asyncpg://user:password@host:port/dbname
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api_gateway.config import settings

# ─── Engine and session factory ───────────────────────────────────────────────

# create_async_engine is module-level — one engine per process.
# pool_pre_ping=True verifies connections before use (handles stale connections
# after database restarts without raising errors to the application).
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug,  # log SQL statements in debug mode
)

# async_sessionmaker replaces the older sessionmaker for async usage.
# expire_on_commit=False prevents lazy-loading errors after commit —
# attributes remain accessible without an active session.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ─── Dependency ───────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yield an async database session per request.

    Opens a session at the start of the request, yields it to the
    endpoint, and closes it after the response is sent — even if an
    exception occurred. This guarantees no connection leaks.

    The session is NOT committed here — each endpoint must call
    await db.commit() explicitly. This makes transaction boundaries
    visible in the business logic, not hidden in infrastructure.

    Usage:
        @router.post("/commands")
        async def create_command(db: DbSession) -> ...:
            db.add(entry)
            await db.commit()
    """
    async with AsyncSessionLocal() as session:
        yield session


# ─── Type alias ───────────────────────────────────────────────────────────────

# Annotated type for use in router function signatures.
# FastAPI reads the Depends() and injects the session automatically.
#
# Usage: async def my_endpoint(db: DbSession) -> ...
DbSession = Annotated[AsyncSession, Depends(get_db)]
