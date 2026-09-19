"""Alert-manager fixtures: the alarm tables in SQLite and a processor over them."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from alarm_factories import Recorder
from alert_manager.models import Base
from alert_manager.processor import AlarmProcessor
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

CLEAR_HOLD_S = 0.02


@pytest_asyncio.fixture
async def sessions(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """
    A SQLite file, so each session has a connection of its own as with PostgreSQL.

    An in-memory database shares one connection between sessions: a reader closing its
    session would roll back a writer's transaction in flight.
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{(tmp_path / 'alarms.db').as_posix()}"
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def recorder() -> Recorder:
    return Recorder()


@pytest_asyncio.fixture
async def processor(
    sessions: async_sessionmaker[AsyncSession], recorder: Recorder
) -> AsyncIterator[AlarmProcessor]:
    found = AlarmProcessor(sessions, recorder, clear_hold_s=CLEAR_HOLD_S)
    yield found
    await found.close()
