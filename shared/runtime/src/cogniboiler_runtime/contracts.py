"""The JSON payloads of the MQTT contract, typed once.

    alerts/warning, alerts/critical  AlarmConditionMessage  plc-controller -> alert-manager
    alerts/snapshot                  AlarmSnapshotMessage   plc-controller -> alert-manager
    plc/events                       PlcEventMessage        plc-controller -> api-gateway,
                                                            historian
    alarms/changes                   AlarmChangeMessage     alert-manager -> api-gateway,
                                                            historian, opcua-server

Each side used to spell the keys out as string literals, so a rename on one side broke the
other with no failing check. A producer now builds its payload from one of these models and
a consumer parses through it; the round-trip test in this package's tests drives every
producer's output through every consumer.

The contract changes only additively. `parse` reads the fields it knows and ignores any
others, so a field added by a producer reaches a consumer that does not know it yet without
a failure. Encoding keeps the key order the producers always used, so the bytes on the wire
are the same as before the models existed.

What `parse` checks is the contract: every field is present with its JSON type, numbers are
finite and timestamps are whole non-negative epoch milliseconds. What a consumer can store
(text lengths, how far ahead a timestamp may be) is the consumer's own check.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from cogniboiler_runtime.payloads import (
    MAX_JSON_PAYLOAD_BYTES,
    decode_json_object,
    finite_number,
)

SEVERITIES: tuple[str, ...] = ("warning", "critical")
CONDITION_STATES: tuple[str, ...] = ("active", "cleared")
ALARM_STATES: tuple[str, ...] = (
    "ACTIVE_UNACK",
    "ACTIVE_ACK",
    "CLEARED_UNACK",
    "CLEARED",
)


class ContractError(ValueError):
    """A payload that breaks its topic's contract; `reason` labels the rejection."""

    def __init__(self, message: str, *, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = reason


def decode_payload(
    raw: bytes | bytearray | Mapping[str, Any],
    *,
    max_bytes: int = MAX_JSON_PAYLOAD_BYTES,
) -> Mapping[str, Any]:
    """The JSON object of a payload; a mapping that is already decoded passes through."""
    if isinstance(raw, Mapping):
        return raw
    if len(raw) > max_bytes:
        raise ContractError(
            f"payload too large: {len(raw)} bytes > {max_bytes}", reason="too_large"
        )
    payload = decode_json_object(raw, max_bytes=max_bytes)
    if payload is None:
        raise ContractError("payload is not JSON, or not a JSON object")
    return payload


def _encode(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _Fields:
    """Typed reads of one JSON object; every failure names the field."""

    def __init__(self, payload: Mapping[str, Any], prefix: str = "") -> None:
        self._payload = payload
        self._prefix = prefix

    def _name(self, field: str) -> str:
        return self._prefix + field

    def require(self, fields: Iterable[str]) -> None:
        missing = sorted(self._name(f) for f in fields if f not in self._payload)
        if missing:
            raise ContractError(f"missing fields: {', '.join(missing)}")

    def text(self, field: str, default: str | None = None) -> str:
        value = self._payload.get(field, default)
        if not isinstance(value, str):
            raise ContractError(f"{self._name(field)} must be a string")
        return value

    def optional_text(self, field: str) -> str | None:
        if self._payload.get(field) is None:
            return None
        return self.text(field)

    def choice(
        self, field: str, allowed: Collection[str], default: str | None = None
    ) -> str:
        value = self.text(field, default)
        if value not in allowed:
            raise ContractError(f"unknown {self._name(field)} {value!r}")
        return value

    def optional_choice(self, field: str, allowed: Collection[str]) -> str | None:
        if self._payload.get(field) is None:
            return None
        return self.choice(field, allowed)

    def number(self, field: str) -> float:
        value = self._payload.get(field)
        number = finite_number(value)
        if number is None:
            numeric = isinstance(value, int | float) and not isinstance(value, bool)
            kind = "a finite number" if numeric else "a number"
            raise ContractError(f"{self._name(field)} must be {kind}")
        return number

    def optional_number(self, field: str) -> float | None:
        if self._payload.get(field) is None:
            return None
        return self.number(field)

    def integer(self, field: str) -> int:
        value = self._payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{self._name(field)} must be an integer")
        return value

    def ms(self, field: str) -> int:
        """UTC epoch milliseconds: a whole, non-negative number."""
        number = self.number(field)
        if not number.is_integer() or number < 0:
            raise ContractError(
                f"{self._name(field)} {self._payload.get(field)!r} is not a plausible time"
            )
        return int(number)

    def optional_ms(self, field: str) -> int | None:
        if self._payload.get(field) is None:
            return None
        return self.ms(field)

    def object(self, field: str) -> Mapping[str, Any]:
        value = self._payload.get(field)
        if not isinstance(value, Mapping):
            raise ContractError(f"{self._name(field)} must be an object")
        return value

    def non_empty(self, *fields: str) -> None:
        if any(not self.text(field) for field in fields):
            names = " and ".join(self._name(field) for field in fields)
            raise ContractError(f"{names} must not be empty")


@dataclass(frozen=True, slots=True)
class AlarmConditionMessage:
    """alerts/warning, alerts/critical: a condition became active or ended at its source.

    `alarm_id` is the source's own `key:timestamp_ms` and no consumer reads it. A payload
    from before alarm lifecycles has no `key`, `state`, `direction`, `unit`, `action`,
    `alarm_id` or `raised_at_ms`; it parses with `key` empty and `state` "active", and the
    consumer decides what an empty key means.
    """

    alarm_id: str
    key: str
    state: str
    source_service: str
    severity: str
    parameter: str
    direction: str
    unit: str
    value: float
    threshold: float
    action: str
    message: str
    raised_at_ms: int | None
    timestamp_ms: int

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "source_service",
        "severity",
        "parameter",
        "value",
        "threshold",
        "message",
        "timestamp_ms",
    )

    @property
    def active(self) -> bool:
        return self.state == "active"

    def to_payload(self) -> dict[str, Any]:
        return {
            "alarm_id": self.alarm_id,
            "key": self.key,
            "state": self.state,
            "source_service": self.source_service,
            "severity": self.severity,
            "parameter": self.parameter,
            "direction": self.direction,
            "unit": self.unit,
            "value": self.value,
            "threshold": self.threshold,
            "action": self.action,
            "message": self.message,
            "raised_at_ms": self.raised_at_ms,
            "timestamp_ms": self.timestamp_ms,
        }

    def encode(self) -> bytes:
        return _encode(self.to_payload())

    @classmethod
    def parse(cls, raw: bytes | bytearray | Mapping[str, Any]) -> Self:
        fields = _Fields(decode_payload(raw))
        fields.require(cls.REQUIRED)
        severity = fields.choice("severity", SEVERITIES)
        fields.non_empty("source_service", "parameter")
        state = fields.choice("state", CONDITION_STATES, default="active")
        return cls(
            alarm_id=fields.text("alarm_id", default=""),
            key=fields.text("key", default=""),
            state=state,
            source_service=fields.text("source_service"),
            severity=severity,
            parameter=fields.text("parameter"),
            direction=fields.text("direction", default=""),
            unit=fields.text("unit", default=""),
            value=fields.number("value"),
            threshold=fields.number("threshold"),
            action=fields.text("action", default="warn"),
            message=fields.text("message"),
            raised_at_ms=fields.optional_ms("raised_at_ms"),
            timestamp_ms=fields.ms("timestamp_ms"),
        )


@dataclass(frozen=True, slots=True)
class AlarmSnapshotMessage:
    """alerts/snapshot: every condition key a source reports active at one moment."""

    source_service: str
    active_keys: tuple[str, ...]
    timestamp_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_service": self.source_service,
            "active_keys": list(self.active_keys),
            "timestamp_ms": self.timestamp_ms,
        }

    def encode(self) -> bytes:
        return _encode(self.to_payload())

    @classmethod
    def parse(cls, raw: bytes | bytearray | Mapping[str, Any]) -> Self:
        payload = decode_payload(raw)
        fields = _Fields(payload)
        keys = payload.get("active_keys")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ContractError("active_keys must be a list of strings")
        source = fields.text("source_service", default="")
        if not source:
            raise ContractError("source_service must not be empty")
        return cls(
            source_service=source,
            active_keys=tuple(keys),
            timestamp_ms=fields.ms("timestamp_ms"),
        )


@dataclass(frozen=True, slots=True)
class PlcEventMessage:
    """plc/events: something the PLC did or refused, and who asked for it.

    `operator_id` is empty for what the PLC did on its own (an interlock trip).
    """

    event_id: str
    kind: str
    source_service: str
    operator_id: str
    detail: Mapping[str, Any]
    timestamp_ms: int

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "event_id",
        "kind",
        "source_service",
        "operator_id",
        "detail",
        "timestamp_ms",
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "source_service": self.source_service,
            "operator_id": self.operator_id,
            "detail": dict(self.detail),
            "timestamp_ms": self.timestamp_ms,
        }

    def encode(self) -> bytes:
        return _encode(self.to_payload())

    @classmethod
    def parse(cls, raw: bytes | bytearray | Mapping[str, Any]) -> Self:
        fields = _Fields(decode_payload(raw))
        fields.require(cls.REQUIRED)
        fields.non_empty("kind")
        return cls(
            event_id=fields.text("event_id"),
            kind=fields.text("kind"),
            source_service=fields.text("source_service"),
            operator_id=fields.text("operator_id"),
            detail=dict(fields.object("detail")),
            timestamp_ms=fields.ms("timestamp_ms"),
        )


@dataclass(frozen=True, slots=True)
class AlarmRecord:
    """The alarm inside an alarms/changes message, as alert-manager stores it."""

    id: int
    key: str
    source_service: str
    parameter: str
    severity: str
    direction: str
    unit: str
    state: str
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

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "id",
        "key",
        "source_service",
        "parameter",
        "severity",
        "direction",
        "unit",
        "state",
        "message",
        "action",
        "topic",
        "value",
        "threshold",
        "raised_at_ms",
        "cleared_at_ms",
        "acknowledged_at_ms",
        "acknowledged_by",
        "ack_comment",
        "occurrence_count",
        "updated_at_ms",
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "source_service": self.source_service,
            "parameter": self.parameter,
            "severity": self.severity,
            "direction": self.direction,
            "unit": self.unit,
            "state": self.state,
            "message": self.message,
            "action": self.action,
            "topic": self.topic,
            "value": self.value,
            "threshold": self.threshold,
            "raised_at_ms": self.raised_at_ms,
            "cleared_at_ms": self.cleared_at_ms,
            "acknowledged_at_ms": self.acknowledged_at_ms,
            "acknowledged_by": self.acknowledged_by,
            "ack_comment": self.ack_comment,
            "occurrence_count": self.occurrence_count,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> Self:
        fields = _Fields(payload, prefix="alarm.")
        fields.require(cls.REQUIRED)
        return cls(
            id=fields.integer("id"),
            key=fields.text("key"),
            source_service=fields.text("source_service"),
            parameter=fields.text("parameter"),
            severity=fields.choice("severity", SEVERITIES),
            direction=fields.text("direction"),
            unit=fields.text("unit"),
            state=fields.choice("state", ALARM_STATES),
            message=fields.text("message"),
            action=fields.text("action"),
            topic=fields.text("topic"),
            value=fields.number("value"),
            threshold=fields.number("threshold"),
            raised_at_ms=fields.ms("raised_at_ms"),
            cleared_at_ms=fields.optional_ms("cleared_at_ms"),
            acknowledged_at_ms=fields.optional_ms("acknowledged_at_ms"),
            acknowledged_by=fields.optional_text("acknowledged_by"),
            ack_comment=fields.optional_text("ack_comment"),
            occurrence_count=fields.integer("occurrence_count"),
            updated_at_ms=fields.ms("updated_at_ms"),
        )


@dataclass(frozen=True, slots=True)
class AlarmTransitionRecord:
    """The state change inside an alarms/changes message; `from_state` is None when the
    alarm opened."""

    id: int
    alarm_id: int
    from_state: str | None
    to_state: str
    at_ms: int
    actor: str
    comment: str | None
    value: float | None

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "id",
        "alarm_id",
        "from_state",
        "to_state",
        "at_ms",
        "actor",
        "comment",
        "value",
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alarm_id": self.alarm_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "at_ms": self.at_ms,
            "actor": self.actor,
            "comment": self.comment,
            "value": self.value,
        }

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> Self:
        fields = _Fields(payload, prefix="transition.")
        fields.require(cls.REQUIRED)
        return cls(
            id=fields.integer("id"),
            alarm_id=fields.integer("alarm_id"),
            from_state=fields.optional_choice("from_state", ALARM_STATES),
            to_state=fields.choice("to_state", ALARM_STATES),
            at_ms=fields.ms("at_ms"),
            actor=fields.text("actor"),
            comment=fields.optional_text("comment"),
            value=fields.optional_number("value"),
        )


@dataclass(frozen=True, slots=True)
class AlarmChangeMessage:
    """alarms/changes: an alarm right after a state change, with that transition."""

    alarm: AlarmRecord
    transition: AlarmTransitionRecord
    timestamp_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "alarm": self.alarm.to_payload(),
            "transition": self.transition.to_payload(),
            "timestamp_ms": self.timestamp_ms,
        }

    def encode(self) -> bytes:
        return _encode(self.to_payload())

    @classmethod
    def parse(cls, raw: bytes | bytearray | Mapping[str, Any]) -> Self:
        fields = _Fields(decode_payload(raw))
        fields.require(("alarm", "transition", "timestamp_ms"))
        return cls(
            alarm=AlarmRecord.parse(fields.object("alarm")),
            transition=AlarmTransitionRecord.parse(fields.object("transition")),
            timestamp_ms=fields.ms("timestamp_ms"),
        )
