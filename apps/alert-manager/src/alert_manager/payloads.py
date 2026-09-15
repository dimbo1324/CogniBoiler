"""
MQTT payloads the alert manager consumes and produces.

Consumed, from the PLC:
    alerts/warning, alerts/critical  a condition became active or ended
    alerts/snapshot                  every condition key the source reports active

A condition payload without "key" or "state" — the format before alarm lifecycles — is
still accepted: the key is derived from source, parameter and severity, and the condition
counts as active.

Produced:
    alarms/changes                   the alarm after a state change, with the transition
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from alert_manager.views import AlarmView, TransitionView

SUBSCRIBE_TOPIC: str = "alerts/#"
TOPIC_SNAPSHOT: str = "alerts/snapshot"
TOPIC_CHANGES: str = "alarms/changes"

SEVERITIES: frozenset[str] = frozenset({"warning", "critical"})
REQUIRED_CONDITION_FIELDS: frozenset[str] = frozenset(
    {
        "source_service",
        "severity",
        "parameter",
        "value",
        "threshold",
        "message",
        "timestamp_ms",
    }
)


class PayloadError(ValueError):
    """A payload that is not a valid alarm message."""


@dataclass(frozen=True)
class ConditionReport:
    """A condition that started or ended at its source."""

    key: str
    source_service: str
    parameter: str
    severity: str
    direction: str
    unit: str
    value: float
    threshold: float
    action: str
    message: str
    topic: str
    active: bool
    timestamp_ms: int


@dataclass(frozen=True)
class SnapshotReport:
    """The conditions a source reports active at one moment."""

    source_service: str
    active_keys: frozenset[str]
    timestamp_ms: int


def _decode(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PayloadError(f"not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PayloadError("payload is not a JSON object")
    return payload


def _text(
    payload: dict[str, Any], field: str, default: str = "", limit: int = 200
) -> str:
    value = payload.get(field, default)
    if not isinstance(value, str):
        raise PayloadError(f"{field} must be a string")
    return value[:limit]


def _number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PayloadError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PayloadError(f"{field} must be finite")
    return number


def parse_condition(topic: str, raw: bytes) -> ConditionReport:
    """Decode an alerts/warning or alerts/critical payload."""
    payload = _decode(raw)
    missing = REQUIRED_CONDITION_FIELDS - payload.keys()
    if missing:
        raise PayloadError(f"missing fields: {', '.join(sorted(missing))}")

    severity = _text(payload, "severity")
    if severity not in SEVERITIES:
        raise PayloadError(f"unknown severity {severity!r}")
    source = _text(payload, "source_service", limit=64)
    parameter = _text(payload, "parameter", limit=128)
    if not source or not parameter:
        raise PayloadError("source_service and parameter must not be empty")

    state = _text(payload, "state", default="active")
    if state not in ("active", "cleared"):
        raise PayloadError(f"unknown state {state!r}")

    return ConditionReport(
        key=_text(payload, "key") or f"{source}:{parameter}:{severity}",
        source_service=source,
        parameter=parameter,
        severity=severity,
        direction=_text(payload, "direction", limit=8),
        unit=_text(payload, "unit", limit=16),
        value=_number(payload, "value"),
        threshold=_number(payload, "threshold"),
        action=_text(payload, "action", default="warn", limit=32),
        message=_text(payload, "message", limit=2000),
        topic=topic[:128],
        active=state == "active",
        timestamp_ms=int(_number(payload, "timestamp_ms")),
    )


def parse_snapshot(raw: bytes) -> SnapshotReport:
    """Decode an alerts/snapshot payload."""
    payload = _decode(raw)
    keys = payload.get("active_keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise PayloadError("active_keys must be a list of strings")
    source = _text(payload, "source_service", limit=64)
    if not source:
        raise PayloadError("source_service must not be empty")
    return SnapshotReport(
        source_service=source,
        active_keys=frozenset(keys),
        timestamp_ms=int(_number(payload, "timestamp_ms")),
    )


def change_payload(alarm: AlarmView, transition: TransitionView) -> bytes:
    """Encode an alarms/changes message."""
    return json.dumps(
        {
            "alarm": alarm.to_dict(),
            "transition": transition.to_dict(),
            "timestamp_ms": transition.at_ms,
        }
    ).encode("utf-8")
