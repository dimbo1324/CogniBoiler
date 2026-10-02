"""
MQTT payloads the alert manager consumes and produces.

Consumed, from the PLC:
    alerts/warning, alerts/critical  a condition became active or ended
    alerts/snapshot                  every condition key the source reports active

The shapes are the shared contract (`cogniboiler_runtime.contracts`); what is checked here
is what this service stores. A condition payload without "key" or "state" — the format
before alarm lifecycles — is still accepted: the key is derived from source, parameter and
severity, and the condition counts as active.

Produced:
    alarms/changes                   the alarm after a state change, with the transition
"""

from __future__ import annotations

from dataclasses import dataclass

from cogniboiler_runtime import now_ms
from cogniboiler_runtime.contracts import (
    AlarmChangeMessage,
    AlarmConditionMessage,
    AlarmSnapshotMessage,
    ContractError,
)

from alert_manager.views import AlarmView, TransitionView

MAX_SNAPSHOT_KEYS: int = 1000
MAX_KEY_LENGTH: int = 200
MAX_FUTURE_SKEW_MS: int = 86_400_000

# The contract's rejection; `reason` labels the rejected-message metric.
PayloadError = ContractError


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


def _plausible(timestamp_ms: int) -> int:
    """No further ahead of this clock than a day: a source's clock may drift, not jump."""
    if timestamp_ms > now_ms() + MAX_FUTURE_SKEW_MS:
        raise PayloadError(f"timestamp_ms {timestamp_ms!r} is not a plausible time")
    return timestamp_ms


def parse_condition(topic: str, raw: bytes) -> ConditionReport:
    """Decode an alerts/warning or alerts/critical payload."""
    message = AlarmConditionMessage.parse(raw)
    source = message.source_service[:64]
    parameter = message.parameter[:128]
    return ConditionReport(
        key=message.key[:MAX_KEY_LENGTH] or f"{source}:{parameter}:{message.severity}",
        source_service=source,
        parameter=parameter,
        severity=message.severity,
        direction=message.direction[:8],
        unit=message.unit[:16],
        value=message.value,
        threshold=message.threshold,
        action=message.action[:32],
        message=message.message[:2000],
        topic=topic[:128],
        active=message.active,
        timestamp_ms=_plausible(message.timestamp_ms),
    )


def parse_snapshot(raw: bytes) -> SnapshotReport:
    """Decode an alerts/snapshot payload."""
    message = AlarmSnapshotMessage.parse(raw)
    active_keys = frozenset(message.active_keys)
    if len(active_keys) > MAX_SNAPSHOT_KEYS:
        raise PayloadError(f"too many active_keys: {len(active_keys)}")
    if any(len(key) > MAX_KEY_LENGTH for key in active_keys):
        raise PayloadError(f"an active key is too long (> {MAX_KEY_LENGTH})")
    return SnapshotReport(
        source_service=message.source_service[:64],
        active_keys=active_keys,
        timestamp_ms=_plausible(message.timestamp_ms),
    )


def change_payload(alarm: AlarmView, transition: TransitionView) -> bytes:
    """Encode an alarms/changes message."""
    return AlarmChangeMessage(
        alarm=alarm.to_record(),
        transition=transition.to_record(),
        timestamp_ms=transition.at_ms,
    ).encode()
