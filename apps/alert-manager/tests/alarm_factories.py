"""Builders of condition reports, snapshots and payloads for alert-manager tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from alert_manager.lifecycle import AlarmState
from alert_manager.payloads import ConditionReport, SnapshotReport
from alert_manager.processor import AlarmProcessor, AlarmQuery
from alert_manager.views import AlarmView, TransitionView

SOURCE = "plc-controller"


def condition(
    parameter: str = "water_level_m",
    *,
    severity: str = "critical",
    active: bool = True,
    value: float = 3.1,
    timestamp_ms: int = 1_000,
    source: str = SOURCE,
    direction: str = "low",
) -> ConditionReport:
    return ConditionReport(
        key=f"{source}:{parameter}:{direction}:{severity}",
        source_service=source,
        parameter=parameter,
        severity=severity,
        direction=direction,
        unit="m",
        value=value,
        threshold=3.5,
        action="trip" if severity == "critical" else "warn",
        message=f"{parameter} {direction}",
        topic=f"alerts/{severity}",
        active=active,
        timestamp_ms=timestamp_ms,
    )


def snapshot(
    *keys: str, timestamp_ms: int = 5_000, source: str = SOURCE
) -> SnapshotReport:
    return SnapshotReport(
        source_service=source, active_keys=frozenset(keys), timestamp_ms=timestamp_ms
    )


def condition_payload(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "key": "plc-controller:water_level_m:low:critical",
        "state": "active",
        "source_service": SOURCE,
        "severity": "critical",
        "parameter": "water_level_m",
        "direction": "low",
        "unit": "m",
        "value": 3.1,
        "threshold": 3.5,
        "action": "trip",
        "message": "Drum level low-low",
        "timestamp_ms": 1_700_000_000_000,
    }
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return json.dumps(payload).encode("utf-8")


class Recorder:
    """A ChangeListener that keeps every change it is told about."""

    def __init__(self) -> None:
        self.changes: list[tuple[AlarmView, TransitionView]] = []

    def alarm_changed(self, alarm: AlarmView, transition: TransitionView) -> None:
        self.changes.append((alarm, transition))

    @property
    def states(self) -> list[tuple[str | None, str]]:
        return [
            (t.from_state.value if t.from_state else None, t.to_state.value)
            for _, t in self.changes
        ]


async def wait_for_state(
    processor: AlarmProcessor, alarm_id: int, state: AlarmState
) -> AlarmView:
    """The alarm once it has reached a state; fails after five seconds."""
    async with asyncio.timeout(5.0):
        while True:
            alarm, _ = await processor.get_alarm(alarm_id)
            if alarm.state is state:
                return alarm
            await asyncio.sleep(0.005)


async def only_alarm(processor: AlarmProcessor) -> AlarmView:
    alarms, total = await processor.list_alarms(AlarmQuery())
    assert total == 1, alarms
    return alarms[0]
