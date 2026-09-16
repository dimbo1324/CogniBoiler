"""
Points for everything the historian records besides boiler and turbine values.

    plant_status         tags scenario; emissions, condenser, equipment health, valve
                         commands and positions, performance (KPIs), simulation speed,
                         time and run id, fault and instrument-quality counts; the active
                         fault labels as the string field fault_labels
    simulation_events    tags kind (scenario_loaded | fault_injected | fault_cleared);
                         fields scenario, run_id, label, text — Grafana annotations
    alarm_changes        tags severity, parameter, state; fields alarm_id, value,
                         threshold, actor, message
    plc_events           tags kind; fields operator_id, detail (JSON), text
    service_availability tags service; field online (1 or 0)
    historian_stats      counters of the historian itself

Numeric fields are floats so one-minute aggregation treats every series alike; text is
kept only in fields the aggregation skips.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cogniboiler_pb2 as pb
from influxdb_client.domain.write_precision import WritePrecision

from historian.writer import PointLike, add_numeric_fields, new_point, timestamp_ns

MEASUREMENT_PLANT: str = "plant_status"
MEASUREMENT_SIMULATION_EVENTS: str = "simulation_events"
MEASUREMENT_ALARM_CHANGES: str = "alarm_changes"
MEASUREMENT_PLC_EVENTS: str = "plc_events"
MEASUREMENT_AVAILABILITY: str = "service_availability"
MEASUREMENT_STATS: str = "historian_stats"

# Measurements whose float fields are downsampled into the aggregate bucket.
AGGREGATED_MEASUREMENTS: tuple[str, ...] = (
    "boiler_sensors",
    "turbine_sensors",
    MEASUREMENT_PLANT,
)

_TEXT_LIMIT = 1000


def now_ms() -> int:
    return int(time.time() * 1000)


def _text(value: object, limit: int = _TEXT_LIMIT) -> str:
    return str(value)[:limit] if value is not None else ""


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_plant_point(msg: pb.PlantStatusMsg) -> PointLike:
    timestamp_ms = msg.timestamp_ms or now_ms()
    scenario = msg.simulation.scenario
    point = new_point(MEASUREMENT_PLANT)
    if scenario:
        point = point.tag("scenario", scenario)
    point = add_numeric_fields(point, msg.emissions)
    point = add_numeric_fields(point, msg.condenser, prefix="condenser_")
    point = add_numeric_fields(point, msg.health)
    point = add_numeric_fields(point, msg.actuators)
    point = add_numeric_fields(point, msg.performance)
    qualities = [sensor.quality for sensor in msg.sensors]
    point = (
        point.field("speed_factor", float(msg.simulation.speed_factor))
        .field("simulation_time_s", float(msg.simulation.simulation_time_s))
        .field("run_id", float(msg.simulation.run_id))
        .field(
            "paused",
            1.0
            if msg.simulation.run_state == pb.SimulationRunState.SIMULATION_PAUSED
            else 0.0,
        )
        .field("fault_count", float(len(msg.active_faults)))
        .field(
            "uncertain_sensor_count", float(qualities.count(pb.SensorQuality.UNCERTAIN))
        )
        .field("bad_sensor_count", float(qualities.count(pb.SensorQuality.BAD)))
        .field("fault_labels", ",".join(sorted(f.label for f in msg.active_faults)))
    )
    return point.time(timestamp_ns(timestamp_ms), WritePrecision.NS)


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    kind: str
    scenario: str
    run_id: int
    label: str
    text: str
    timestamp_ms: int


class RunLabels:
    """
    The scenario and active faults of the plant, and the changes between messages.

    The first message only establishes the state; a restarted historian does not report
    the faults that were already active as new.
    """

    def __init__(self) -> None:
        self._known = False
        self.scenario: str | None = None
        self.run_id: int | None = None
        self._faults: dict[str, str] = {}

    def update(self, msg: pb.PlantStatusMsg) -> list[SimulationEvent]:
        timestamp_ms = msg.timestamp_ms or now_ms()
        scenario = msg.simulation.scenario or None
        run_id = int(msg.simulation.run_id)
        faults = {fault.fault_id: fault.label for fault in msg.active_faults}
        events: list[SimulationEvent] = []
        if self._known:
            if run_id != self.run_id:
                events.append(
                    SimulationEvent(
                        "scenario_loaded",
                        scenario or "",
                        run_id,
                        scenario or "",
                        f"Scenario {scenario} loaded (run {run_id})",
                        timestamp_ms,
                    )
                )
            for fault_id, label in faults.items():
                if fault_id not in self._faults:
                    events.append(
                        SimulationEvent(
                            "fault_injected",
                            scenario or "",
                            run_id,
                            label,
                            f"Fault injected: {label}",
                            timestamp_ms,
                        )
                    )
            if run_id == self.run_id:
                for fault_id, label in self._faults.items():
                    if fault_id not in faults:
                        events.append(
                            SimulationEvent(
                                "fault_cleared",
                                scenario or "",
                                run_id,
                                label,
                                f"Fault cleared: {label}",
                                timestamp_ms,
                            )
                        )
        self._known = True
        self.scenario = scenario
        self.run_id = run_id
        self._faults = faults
        return events


def build_simulation_event_point(event: SimulationEvent) -> PointLike:
    return (
        new_point(MEASUREMENT_SIMULATION_EVENTS)
        .tag("kind", event.kind)
        .field("scenario", event.scenario)
        .field("run_id", float(event.run_id))
        .field("label", event.label)
        .field("text", event.text)
        .time(timestamp_ns(event.timestamp_ms), WritePrecision.NS)
    )


def build_alarm_change_point(payload: Mapping[str, Any]) -> PointLike | None:
    """An alarms/changes message; None if it lacks the alarm or its transition."""
    alarm = payload.get("alarm")
    transition = payload.get("transition")
    if not isinstance(alarm, Mapping) or not isinstance(transition, Mapping):
        return None
    at_ms = transition.get("at_ms")
    if isinstance(at_ms, bool) or not isinstance(at_ms, int):
        return None
    to_state = _text(transition.get("to_state"), 16)
    point = (
        new_point(MEASUREMENT_ALARM_CHANGES)
        .tag("severity", _text(alarm.get("severity"), 16) or "unknown")
        .tag("parameter", _text(alarm.get("parameter"), 128) or "unknown")
        .tag("state", to_state or "unknown")
        .field("alarm_id", _finite(alarm.get("id")) or 0.0)
        .field("actor", _text(transition.get("actor"), 128))
        .field("message", _text(alarm.get("message")))
        .field(
            "text",
            f"{_text(alarm.get('severity'))} {to_state}: {_text(alarm.get('message'))}",
        )
    )
    for name in ("value", "threshold"):
        number = _finite(alarm.get(name))
        if number is not None:
            point = point.field(name, number)
    return point.time(timestamp_ns(at_ms), WritePrecision.NS)


def build_plc_event_point(payload: Mapping[str, Any]) -> PointLike | None:
    """A plc/events message; None if it has no kind or timestamp."""
    kind = payload.get("kind")
    timestamp_ms = payload.get("timestamp_ms")
    if not isinstance(kind, str) or isinstance(timestamp_ms, bool):
        return None
    if not isinstance(timestamp_ms, int):
        return None
    operator = _text(payload.get("operator_id"), 128)
    detail = json.dumps(payload.get("detail") or {}, sort_keys=True)[:_TEXT_LIMIT]
    return (
        new_point(MEASUREMENT_PLC_EVENTS)
        .tag("kind", kind[:32])
        .field("operator_id", operator)
        .field("detail", detail)
        .field("text", f"PLC {kind}" + (f" by {operator}" if operator else ""))
        .time(timestamp_ns(timestamp_ms), WritePrecision.NS)
    )


def build_availability_point(service: str, payload: bytes) -> PointLike | None:
    state = payload.decode("utf-8", errors="replace").strip().lower()
    if state not in ("online", "offline"):
        return None
    return (
        new_point(MEASUREMENT_AVAILABILITY)
        .tag("service", service[:64])
        .field("online", 1.0 if state == "online" else 0.0)
        .time(timestamp_ns(now_ms()), WritePrecision.NS)
    )


def build_stats_point(stats: Mapping[str, int], writer_errors: int) -> PointLike:
    point = new_point(MEASUREMENT_STATS).tag("service", "historian")
    for name, value in stats.items():
        point = point.field(name, float(value))
    return point.field("writer_errors", float(writer_errors)).time(
        timestamp_ns(now_ms()), WritePrecision.NS
    )
