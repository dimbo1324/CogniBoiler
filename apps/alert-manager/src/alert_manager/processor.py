"""
Alarm processing: condition reports become alarms with a lifecycle.

Everything that changes alarm state — conditions starting and ending, source snapshots,
acknowledgements — runs one at a time under a lock, so the database never sees two writers
racing on one alarm. A condition that ends is cleared only after it has stayed normal for
a short hold time: a signal flickering across its limit keeps one alarm open instead of
producing a burst of them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from cogniboiler_runtime import now_ms
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alert_manager.lifecycle import (
    ACTIVE_STATES,
    UNACKNOWLEDGED_STATES,
    AlarmState,
    after_acknowledge,
    after_condition_active,
    after_condition_cleared,
)
from alert_manager.metrics import SNAPSHOT_UNMATCHED_KEYS, TRANSITIONS
from alert_manager.models import AlarmEvent, AlarmTransition
from alert_manager.payloads import ConditionReport, SnapshotReport
from alert_manager.queries import AlarmNotFoundError, check_alarm_id
from alert_manager.views import AlarmView, TransitionView

logger = logging.getLogger(__name__)

CLEAR_HOLD_S: float = 3.0
MAX_COMMENT_LENGTH: int = 500
MAX_OPERATOR_LENGTH: int = 128

_ACTIVE = [state.value for state in ACTIVE_STATES]
_UNACKNOWLEDGED = [state.value for state in UNACKNOWLEDGED_STATES]


class ChangeListener(Protocol):
    """Told about every alarm state change after it is committed."""

    def alarm_changed(self, alarm: AlarmView, transition: TransitionView) -> None: ...


class InvalidOperatorError(ValueError):
    """An acknowledgement that names no operator."""


def _operator(operator_id: str) -> str:
    operator = operator_id.strip()[:MAX_OPERATOR_LENGTH]
    if not operator:
        raise InvalidOperatorError("operator_id must not be empty")
    return operator


class AlarmProcessor:
    """Owns alarm state: applies reports and acknowledgements, one at a time.

    Reads live in `alert_manager.queries`.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        listener: ChangeListener | None = None,
        *,
        clear_hold_s: float = CLEAR_HOLD_S,
    ) -> None:
        self._sessions = sessions
        self._listener = listener
        self._clear_hold_s = clear_hold_s
        self._lock = asyncio.Lock()
        self._pending_clears: dict[str, asyncio.Task[None]] = {}
        self._unmatched_keys: dict[str, frozenset[str]] = {}

    async def close(self) -> None:
        """Cancel clears still waiting out their hold time."""
        tasks = list(self._pending_clears.values())
        self._pending_clears.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def pending_clears(self) -> frozenset[str]:
        """Keys of the clears still waiting out their hold time."""
        return frozenset(self._pending_clears)

    async def drain(self) -> None:
        """Wait until every clear waiting out its hold time has run or been dropped."""
        while self._pending_clears:
            await asyncio.gather(*self._pending_clears.values(), return_exceptions=True)

    # ─── Reports from sources ────────────────────────────────────────────────

    async def handle_condition(self, report: ConditionReport) -> None:
        """Apply a condition that started or ended at its source."""
        if not report.active:
            self._schedule_clear(
                report.key, report.value, report.timestamp_ms, report.source_service
            )
            return
        pending = self._pending_clears.pop(report.key, None)
        if pending is not None:
            pending.cancel()
            logger.info(
                "Alarm %s returned within its clear hold; kept open", report.key
            )
        await self._activate(report)

    async def handle_snapshot(self, report: SnapshotReport) -> None:
        """Clear active alarms of a source that no longer reports their condition.

        A key the source lists with no active alarm here means its activation was lost;
        the snapshot carries no details to raise it from, so it is only reported.
        """
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(AlarmEvent).where(
                        AlarmEvent.source_service == report.source_service,
                        AlarmEvent.state.in_(_ACTIVE),
                    )
                )
            ).all()
        self._report_unmatched(report, frozenset(row.key for row in rows))
        for row in rows:
            if row.raised_at_ms > report.timestamp_ms:
                continue
            if row.key in report.active_keys or row.key in self._pending_clears:
                continue
            logger.info(
                "Snapshot of %s no longer lists %s; clearing it",
                report.source_service,
                row.key,
            )
            self._schedule_clear(
                row.key, row.value, report.timestamp_ms, report.source_service
            )

    def _report_unmatched(
        self, report: SnapshotReport, active_keys: frozenset[str]
    ) -> None:
        unmatched = report.active_keys - active_keys
        known = self._unmatched_keys.get(report.source_service, frozenset())
        for key in sorted(unmatched - known):
            SNAPSHOT_UNMATCHED_KEYS.inc()
            logger.warning(
                "Snapshot of %s lists %s as active, but no active alarm exists: "
                "its activation was lost",
                report.source_service,
                key,
            )
        self._unmatched_keys[report.source_service] = unmatched

    # ─── Acknowledgement ─────────────────────────────────────────────────────

    async def acknowledge(
        self, alarm_id: int, operator_id: str, comment: str = ""
    ) -> AlarmView:
        """Acknowledge one alarm.

        Raises InvalidOperatorError, AlarmNotFoundError or LifecycleError.
        """
        operator = _operator(operator_id)
        check_alarm_id(alarm_id)
        note = comment.strip()[:MAX_COMMENT_LENGTH]
        async with self._lock, self._sessions() as session:
            alarm = await session.get(AlarmEvent, alarm_id)
            if alarm is None:
                raise AlarmNotFoundError(f"alarm {alarm_id} does not exist")
            transition = self._acknowledge_row(session, alarm, operator, note)
            await session.flush()
            await session.commit()
            view = self._notify(alarm, transition)
        logger.info("Alarm %d (%s) acknowledged by %s", alarm_id, alarm.key, operator)
        return view

    async def acknowledge_all(
        self, operator_id: str, comment: str = "", severity: str = ""
    ) -> list[AlarmView]:
        """Acknowledge every unacknowledged alarm, optionally of one severity."""
        operator = _operator(operator_id)
        note = comment.strip()[:MAX_COMMENT_LENGTH]
        async with self._lock, self._sessions() as session:
            statement = select(AlarmEvent).where(AlarmEvent.state.in_(_UNACKNOWLEDGED))
            if severity:
                statement = statement.where(AlarmEvent.severity == severity)
            rows = (await session.scalars(statement.order_by(AlarmEvent.id))).all()
            changes = [
                (row, self._acknowledge_row(session, row, operator, note))
                for row in rows
            ]
            await session.flush()
            await session.commit()
            views = [self._notify(alarm, transition) for alarm, transition in changes]
        if views:
            logger.info("%d alarms acknowledged by %s", len(views), operator)
        return views

    # ─── State changes ───────────────────────────────────────────────────────

    async def _activate(self, report: ConditionReport) -> None:
        async with self._lock, self._sessions() as session:
            alarm = await self._open_alarm(session, report.key)
            now = now_ms()
            transition: AlarmTransition | None = None
            if alarm is None:
                alarm = AlarmEvent(
                    key=report.key,
                    source_service=report.source_service,
                    parameter=report.parameter,
                    severity=report.severity,
                    direction=report.direction,
                    unit=report.unit,
                    state=AlarmState.ACTIVE_UNACK.value,
                    message=report.message,
                    action=report.action,
                    topic=report.topic,
                    value=report.value,
                    threshold=report.threshold,
                    raised_at_ms=report.timestamp_ms,
                    occurrence_count=1,
                    updated_at_ms=now,
                )
                session.add(alarm)
                await session.flush()
                transition = self._transition(
                    session,
                    alarm,
                    None,
                    AlarmState.ACTIVE_UNACK,
                    report.timestamp_ms,
                    report.source_service,
                )
            else:
                state = AlarmState(alarm.state)
                new_state = after_condition_active(state)
                alarm.value = report.value
                alarm.message = report.message
                alarm.updated_at_ms = now
                if new_state is not state:
                    alarm.state = new_state.value
                    alarm.cleared_at_ms = None
                    alarm.occurrence_count += 1
                    transition = self._transition(
                        session,
                        alarm,
                        state,
                        new_state,
                        report.timestamp_ms,
                        report.source_service,
                    )
            await session.flush()
            await session.commit()
            if transition is not None:
                self._notify(alarm, transition)

    def _schedule_clear(
        self, key: str, value: float, timestamp_ms: int, actor: str
    ) -> None:
        if key in self._pending_clears:
            return
        task = asyncio.create_task(
            self._clear_after_hold(key, value, timestamp_ms, actor),
            name=f"alarm-clear:{key}",
        )
        self._pending_clears[key] = task

    async def _clear_after_hold(
        self, key: str, value: float, timestamp_ms: int, actor: str
    ) -> None:
        await asyncio.sleep(self._clear_hold_s)
        async with self._lock:
            if self._pending_clears.get(key) is not asyncio.current_task():
                return
            del self._pending_clears[key]
            try:
                await self._apply_clear(key, value, timestamp_ms, actor)
            except Exception:
                logger.exception("Clearing alarm %s failed", key)

    async def _apply_clear(
        self, key: str, value: float, timestamp_ms: int, actor: str
    ) -> None:
        async with self._sessions() as session:
            alarm = await self._open_alarm(session, key)
            if alarm is None:
                return
            state = AlarmState(alarm.state)
            if state not in ACTIVE_STATES:
                return
            new_state = after_condition_cleared(state)
            alarm.state = new_state.value
            alarm.cleared_at_ms = timestamp_ms
            alarm.value = value
            alarm.updated_at_ms = now_ms()
            transition = self._transition(
                session, alarm, state, new_state, timestamp_ms, actor
            )
            await session.flush()
            await session.commit()
            self._notify(alarm, transition)

    def _acknowledge_row(
        self,
        session: AsyncSession,
        alarm: AlarmEvent,
        operator: str,
        comment: str,
    ) -> AlarmTransition:
        state = AlarmState(alarm.state)
        new_state = after_acknowledge(state)
        now = now_ms()
        alarm.state = new_state.value
        alarm.acknowledged_at_ms = now
        alarm.acknowledged_by = operator
        alarm.ack_comment = comment or None
        alarm.updated_at_ms = now
        return self._transition(
            session, alarm, state, new_state, now, operator, comment or None
        )

    @staticmethod
    def _transition(
        session: AsyncSession,
        alarm: AlarmEvent,
        from_state: AlarmState | None,
        to_state: AlarmState,
        at_ms: int,
        actor: str,
        comment: str | None = None,
    ) -> AlarmTransition:
        transition = AlarmTransition(
            alarm_id=alarm.id,
            from_state=from_state.value if from_state is not None else None,
            to_state=to_state.value,
            at_ms=at_ms,
            actor=actor[:MAX_OPERATOR_LENGTH],
            comment=comment,
            value=alarm.value,
        )
        session.add(transition)
        return transition

    def _notify(self, alarm: AlarmEvent, transition: AlarmTransition) -> AlarmView:
        view = AlarmView.from_row(alarm)
        transition_view = TransitionView.from_row(transition)
        logger.info(
            "Alarm %d %s: %s -> %s by %s",
            view.id,
            view.key,
            transition_view.from_state.value if transition_view.from_state else "new",
            transition_view.to_state.value,
            transition_view.actor,
        )
        TRANSITIONS.labels(view.severity, transition_view.to_state.value).inc()
        if self._listener is not None:
            try:
                self._listener.alarm_changed(view, transition_view)
            except Exception:
                # The change is committed; failing here would report it as lost.
                logger.exception("Listener failed on the change of alarm %d", view.id)
        return view

    @staticmethod
    async def _open_alarm(session: AsyncSession, key: str) -> AlarmEvent | None:
        alarm: AlarmEvent | None = await session.scalar(
            select(AlarmEvent).where(
                AlarmEvent.key == key,
                AlarmEvent.state != AlarmState.CLEARED.value,
            )
        )
        return alarm
