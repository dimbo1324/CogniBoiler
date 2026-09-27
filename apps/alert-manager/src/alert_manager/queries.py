"""
Read-only alarm queries: lists, one alarm with its transitions, and the health ping.

Reads take no lock: each runs in a session of its own and sees committed state, so the
lock in `alert_manager.processor` guards only the writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alert_manager.lifecycle import OPEN_STATES, UNACKNOWLEDGED_STATES
from alert_manager.models import AlarmEvent, AlarmTransition
from alert_manager.views import AlarmView, TransitionView

DEFAULT_LIST_LIMIT: int = 100
MAX_LIST_LIMIT: int = 1000
MAX_ALARM_ID: int = 2**31 - 1

_OPEN = [state.value for state in OPEN_STATES]
_UNACKNOWLEDGED = [state.value for state in UNACKNOWLEDGED_STATES]


class AlarmNotFoundError(LookupError):
    """No alarm with the requested id."""


def check_alarm_id(alarm_id: int) -> None:
    """Ids outside the integer column are unknown, not a database error."""
    if not 1 <= alarm_id <= MAX_ALARM_ID:
        raise AlarmNotFoundError(f"alarm {alarm_id} does not exist")


@dataclass(frozen=True)
class AlarmQuery:
    """Filters and paging for alarm lists."""

    open_only: bool = False
    severity: str = ""
    parameter: str = ""
    from_ms: int = 0
    to_ms: int = 0
    limit: int = DEFAULT_LIST_LIMIT
    offset: int = 0


class AlarmQueries:
    """Answers questions about alarms; never changes them."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def ping(self) -> None:
        """Raise if the database cannot be reached."""
        async with self._sessions() as session:
            await session.execute(select(1))

    async def list_alarms(self, query: AlarmQuery) -> tuple[list[AlarmView], int]:
        """Alarms matching a query and how many match in total."""
        limit = min(max(query.limit or DEFAULT_LIST_LIMIT, 1), MAX_LIST_LIMIT)
        offset = max(query.offset, 0)
        conditions: list[ColumnElement[bool]] = []
        if query.open_only:
            conditions.append(AlarmEvent.state.in_(_OPEN))
        if query.severity:
            conditions.append(AlarmEvent.severity == query.severity)
        if query.parameter:
            conditions.append(AlarmEvent.parameter == query.parameter)
        if query.from_ms > 0:
            conditions.append(AlarmEvent.raised_at_ms >= query.from_ms)
        if query.to_ms > 0:
            conditions.append(AlarmEvent.raised_at_ms <= query.to_ms)

        if query.open_only:
            order: tuple[ColumnElement[Any], ...] = (
                case((AlarmEvent.severity == "critical", 0), else_=1),
                case((AlarmEvent.state.in_(_UNACKNOWLEDGED), 0), else_=1),
                AlarmEvent.raised_at_ms.desc(),
                AlarmEvent.id.desc(),
            )
        else:
            order = (AlarmEvent.raised_at_ms.desc(), AlarmEvent.id.desc())

        async with self._sessions() as session:
            total = await session.scalar(
                select(func.count()).select_from(AlarmEvent).where(*conditions)
            )
            rows = (
                await session.scalars(
                    select(AlarmEvent)
                    .where(*conditions)
                    .order_by(*order)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        return [AlarmView.from_row(row) for row in rows], int(total or 0)

    async def get_alarm(self, alarm_id: int) -> tuple[AlarmView, list[TransitionView]]:
        """One alarm with its transitions, oldest first."""
        check_alarm_id(alarm_id)
        async with self._sessions() as session:
            alarm = await session.get(AlarmEvent, alarm_id)
            if alarm is None:
                raise AlarmNotFoundError(f"alarm {alarm_id} does not exist")
            transitions = (
                await session.scalars(
                    select(AlarmTransition)
                    .where(AlarmTransition.alarm_id == alarm_id)
                    .order_by(AlarmTransition.at_ms, AlarmTransition.id)
                )
            ).all()
        return AlarmView.from_row(alarm), [
            TransitionView.from_row(row) for row in transitions
        ]
