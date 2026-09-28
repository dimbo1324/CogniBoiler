"""Mapping of AlarmService protobuf messages to the REST and WebSocket schema."""

from __future__ import annotations

from typing import cast

import cogniboiler_pb2 as pb2

from api_gateway.schemas.ops import (
    AlarmResponse,
    AlarmStateName,
    AlarmTransitionResponse,
)

_STATE_NAMES: dict[int, str] = {
    int(pb2.AlarmState.ALARM_ACTIVE_UNACK): "ACTIVE_UNACK",
    int(pb2.AlarmState.ALARM_ACTIVE_ACK): "ACTIVE_ACK",
    int(pb2.AlarmState.ALARM_CLEARED_UNACK): "CLEARED_UNACK",
    int(pb2.AlarmState.ALARM_CLEARED): "CLEARED",
}
_ACKNOWLEDGED = frozenset({"ACTIVE_ACK", "CLEARED"})
_CLEARED = frozenset({"CLEARED_UNACK", "CLEARED"})


def _state(value: int) -> AlarmStateName:
    return cast(AlarmStateName, _STATE_NAMES.get(int(value), "CLEARED"))


def alarm_from_proto(message: pb2.AlarmMsg) -> AlarmResponse:
    """Map an AlarmService alarm to the REST schema."""
    state = _state(message.state)
    return AlarmResponse(
        alarm_id=str(message.alarm_id),
        id=message.alarm_id,
        key=message.key,
        source_service=message.source_service,
        severity=message.severity,
        parameter=message.parameter,
        direction=message.direction,
        unit=message.unit,
        state=state,
        value=message.value,
        threshold=message.threshold,
        action=message.action,
        message=message.message,
        topic=message.topic,
        occurred_at_ms=message.raised_at_ms,
        raised_at_ms=message.raised_at_ms,
        cleared_at_ms=message.cleared_at_ms or None,
        acknowledged=state in _ACKNOWLEDGED,
        acknowledged_at_ms=message.acknowledged_at_ms or None,
        acknowledged_by=message.acknowledged_by or None,
        ack_comment=message.ack_comment or None,
        cleared=state in _CLEARED,
        occurrence_count=message.occurrence_count,
        updated_at_ms=message.updated_at_ms,
    )


def transition_from_proto(message: pb2.AlarmTransitionMsg) -> AlarmTransitionResponse:
    return AlarmTransitionResponse(
        id=message.transition_id,
        alarm_id=message.alarm_id,
        from_state=(
            _state(message.from_state)
            if message.from_state != pb2.AlarmState.ALARM_STATE_UNSPECIFIED
            else None
        ),
        to_state=_state(message.to_state),
        at_ms=message.at_ms,
        actor=message.actor,
        comment=message.comment or None,
        value=message.value,
    )
