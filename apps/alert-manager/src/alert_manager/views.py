"""Immutable views of alarms and transitions, detached from database sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cogniboiler_runtime.contracts import AlarmRecord, AlarmTransitionRecord

from alert_manager.lifecycle import OPEN_STATES, UNACKNOWLEDGED_STATES, AlarmState
from alert_manager.models import AlarmEvent, AlarmTransition


@dataclass(frozen=True)
class AlarmView:
    """An alarm as read from the database."""

    id: int
    key: str
    source_service: str
    parameter: str
    severity: str
    direction: str
    unit: str
    state: AlarmState
    message: str
    action: str
    topic: str
    value: float
    threshold: float
    raised_at_ms: int
    cleared_at_ms: int | None
    acknowledged_at_ms: int | None
    acknowledged_by: str | None
    ack_comment: str | None
    occurrence_count: int
    updated_at_ms: int

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    @property
    def is_acknowledged(self) -> bool:
        return self.state not in UNACKNOWLEDGED_STATES

    @classmethod
    def from_row(cls, row: AlarmEvent) -> AlarmView:
        return cls(
            id=row.id,
            key=row.key,
            source_service=row.source_service,
            parameter=row.parameter,
            severity=row.severity,
            direction=row.direction,
            unit=row.unit,
            state=AlarmState(row.state),
            message=row.message,
            action=row.action,
            topic=row.topic,
            value=row.value,
            threshold=row.threshold,
            raised_at_ms=row.raised_at_ms,
            cleared_at_ms=row.cleared_at_ms,
            acknowledged_at_ms=row.acknowledged_at_ms,
            acknowledged_by=row.acknowledged_by,
            ack_comment=row.ack_comment,
            occurrence_count=row.occurrence_count,
            updated_at_ms=row.updated_at_ms,
        )

    def to_record(self) -> AlarmRecord:
        """The alarm as the alarms/changes contract carries it."""
        return AlarmRecord(
            id=self.id,
            key=self.key,
            source_service=self.source_service,
            parameter=self.parameter,
            severity=self.severity,
            direction=self.direction,
            unit=self.unit,
            state=self.state.value,
            message=self.message,
            action=self.action,
            topic=self.topic,
            value=self.value,
            threshold=self.threshold,
            raised_at_ms=self.raised_at_ms,
            cleared_at_ms=self.cleared_at_ms,
            acknowledged_at_ms=self.acknowledged_at_ms,
            acknowledged_by=self.acknowledged_by,
            ack_comment=self.ack_comment,
            occurrence_count=self.occurrence_count,
            updated_at_ms=self.updated_at_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_record().to_payload()


@dataclass(frozen=True)
class TransitionView:
    """A state change of an alarm."""

    id: int
    alarm_id: int
    from_state: AlarmState | None
    to_state: AlarmState
    at_ms: int
    actor: str
    comment: str | None
    value: float | None

    @classmethod
    def from_row(cls, row: AlarmTransition) -> TransitionView:
        return cls(
            id=row.id,
            alarm_id=row.alarm_id,
            from_state=AlarmState(row.from_state) if row.from_state else None,
            to_state=AlarmState(row.to_state),
            at_ms=row.at_ms,
            actor=row.actor,
            comment=row.comment,
            value=row.value,
        )

    def to_record(self) -> AlarmTransitionRecord:
        """The transition as the alarms/changes contract carries it."""
        return AlarmTransitionRecord(
            id=self.id,
            alarm_id=self.alarm_id,
            from_state=self.from_state.value if self.from_state else None,
            to_state=self.to_state.value,
            at_ms=self.at_ms,
            actor=self.actor,
            comment=self.comment,
            value=self.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_record().to_payload()
