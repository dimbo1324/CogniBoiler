"""The PLC's status as the contract renders it.

Everything a caller learns about the PLC over gRPC is built here, out of plain values the
service already holds — no reading back from the plant, no side effects. Keeping the
projection apart from the scan loop means a change to the contract is a change to one
file, and the shape of the message can be tested without a PLC, a broker or a plant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cogniboiler_pb2 as pb2

from plc_controller.alarms import AlarmCondition
from plc_controller.commands import CommandSnapshot, Setpoints
from plc_controller.control import ControlTargets, LoopStatus
from plc_controller.safety_limits import SafetyEvent


@dataclass
class SafetySnapshot:
    """The event that latched the E-Stop, kept for as long as the trip stands."""

    timestamp_ms: int
    parameter: str
    value: float
    threshold: float
    level: str
    action: str

    @classmethod
    def from_event(cls, event: SafetyEvent) -> SafetySnapshot:
        return cls(
            timestamp_ms=event.timestamp_ms,
            parameter=event.parameter,
            value=event.value,
            threshold=event.threshold,
            level=event.level.value,
            action=event.action.value,
        )


def setpoints_msg(setpoints: Setpoints) -> pb2.SetpointsMsg:
    return pb2.SetpointsMsg(
        pressure_pa=setpoints.pressure_pa,
        water_level_m=setpoints.water_level_m,
        steam_temp_k=setpoints.steam_temp_k,
        timestamp_ms=setpoints.updated_at_ms,
    )


def command_msg(command: CommandSnapshot) -> pb2.ControlCommandMsg:
    return pb2.ControlCommandMsg(
        fuel_valve=command.fuel_valve,
        feedwater_valve=command.feedwater_valve,
        steam_valve=command.steam_valve,
        spray_valve=command.spray_valve,
        timestamp_ms=command.timestamp_ms,
        source=command.source,
        operator_id=command.operator_id,
    )


def trip_msg(cause: SafetySnapshot | None) -> pb2.SafetyEventMsg:
    """An empty message means no trip stands; the fields are never half-filled."""
    if cause is None:
        return pb2.SafetyEventMsg()
    return pb2.SafetyEventMsg(
        timestamp_ms=cause.timestamp_ms,
        parameter=cause.parameter,
        value=cause.value,
        threshold=cause.threshold,
        level=cause.level,
        action=cause.action,
    )


def condition_msg(condition: AlarmCondition) -> pb2.AlarmConditionMsg:
    return pb2.AlarmConditionMsg(
        key=condition.key,
        parameter=condition.rule.parameter,
        severity=condition.rule.severity.value,
        direction=condition.rule.direction.value,
        value=condition.value,
        threshold=condition.rule.threshold,
        message=condition.message,
        since_ms=condition.since_ms,
    )


def loop_msg(loop: LoopStatus) -> pb2.ControlLoopMsg:
    return pb2.ControlLoopMsg(
        name=loop.name,
        setpoint=loop.setpoint,
        measurement=loop.measurement,
        output=loop.output,
        unit=loop.unit,
    )


def control_status(
    *,
    mode: int,
    emergency_stop_active: bool,
    setpoints: Setpoints,
    latest_command: CommandSnapshot,
    warning_count: int,
    trip_count: int,
    trip_cause: SafetySnapshot | None,
    load_demand_w: float,
    working: ControlTargets | None,
    reset_blockers: Sequence[str],
    conditions: Sequence[AlarmCondition],
    loops: Sequence[LoopStatus],
    run_id: int,
) -> pb2.PLCStatusMsg:
    """The PLC's mode, targets, working setpoints, loops and alarm conditions.

    Before the controller is primed there are no working setpoints: the load setpoint
    then reads as the demand itself rather than as zero, so a console never shows a unit
    ramping from nowhere.
    """
    return pb2.PLCStatusMsg(
        mode=mode,
        emergency_stop_active=emergency_stop_active,
        setpoints=setpoints_msg(setpoints),
        latest_command=command_msg(latest_command),
        warning_count=warning_count,
        trip_count=trip_count,
        active_trip=trip_msg(trip_cause),
        load_demand_w=load_demand_w,
        load_setpoint_w=working.load_w if working is not None else load_demand_w,
        active_setpoints=(
            pb2.SetpointsMsg(
                pressure_pa=working.pressure_pa,
                water_level_m=working.water_level_m,
                steam_temp_k=working.steam_temp_k,
            )
            if working is not None
            else pb2.SetpointsMsg()
        ),
        reset_permitted=emergency_stop_active and not reset_blockers,
        reset_blockers=list(reset_blockers),
        active_conditions=[condition_msg(condition) for condition in conditions],
        loops=[loop_msg(loop) for loop in loops],
        run_id=run_id,
    )
