"""What each upstream message means for the address space: (node id, value) updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cogniboiler_pb2 as pb

from opcua_server.address_space import (
    ACTUATOR_FIELD_TO_NODEID,
    CONDENSER_FIELD_TO_NODEID,
    EMISSIONS_FIELD_TO_NODEID,
    HEALTH_FIELD_TO_NODEID,
    NODEID_ACTIVE_FAULTS,
    NODEID_ALARM_COMMUNICATION,
    NODEID_CRITICAL_ACTIVE_COUNT,
    NODEID_EMERGENCY_STOP_ACTIVE,
    NODEID_INSTRUMENTS_NOT_GOOD,
    NODEID_LEVEL_SETPOINT,
    NODEID_LOAD_DEMAND,
    NODEID_LOAD_SETPOINT,
    NODEID_OPEN_ALARM_COUNT,
    NODEID_OPEN_ALARMS,
    NODEID_PAUSED,
    NODEID_PLC_COMMUNICATION,
    NODEID_PLC_MODE,
    NODEID_PRESSURE_SETPOINT,
    NODEID_RESET_BLOCKERS,
    NODEID_RESET_PERMITTED,
    NODEID_RUN_ID,
    NODEID_SCENARIO,
    NODEID_SIMULATION_TIME,
    NODEID_SPEED_FACTOR,
    NODEID_STEAM_TEMP_SETPOINT,
    NODEID_TRIP_CAUSE,
    NODEID_TRIP_COUNT,
    NODEID_UNACKNOWLEDGED_COUNT,
    NODEID_WARNING_COUNT,
    PERFORMANCE_FIELD_TO_NODEID,
    SENSOR_TO_NODEID,
    InitialValue,
)

Update = tuple[int, InitialValue]

_ALARM_STATES = {
    pb.AlarmState.ALARM_ACTIVE_UNACK: "ACTIVE_UNACK",
    pb.AlarmState.ALARM_ACTIVE_ACK: "ACTIVE_ACK",
    pb.AlarmState.ALARM_CLEARED_UNACK: "CLEARED_UNACK",
    pb.AlarmState.ALARM_CLEARED: "CLEARED",
}


def field_updates(message: Any, mapping: Mapping[str, int]) -> list[Update]:
    """One update per mapped scalar field; the server coerces to the node's type."""
    return [(node_id, getattr(message, field)) for field, node_id in mapping.items()]


def plant_updates(msg: pb.PlantStatusMsg) -> list[Update]:
    simulation = msg.simulation
    not_good = sum(
        1 for sensor in msg.sensors if sensor.quality != pb.SensorQuality.GOOD
    )
    return [
        *field_updates(msg.actuators, ACTUATOR_FIELD_TO_NODEID),
        *field_updates(msg.emissions, EMISSIONS_FIELD_TO_NODEID),
        *field_updates(msg.condenser, CONDENSER_FIELD_TO_NODEID),
        *field_updates(msg.performance, PERFORMANCE_FIELD_TO_NODEID),
        *field_updates(msg.health, HEALTH_FIELD_TO_NODEID),
        (NODEID_SCENARIO, simulation.scenario),
        (NODEID_RUN_ID, int(simulation.run_id)),
        (NODEID_SIMULATION_TIME, simulation.simulation_time_s),
        (NODEID_SPEED_FACTOR, simulation.speed_factor),
        (
            NODEID_PAUSED,
            simulation.run_state == pb.SimulationRunState.SIMULATION_PAUSED,
        ),
        (NODEID_ACTIVE_FAULTS, sorted(fault.label for fault in msg.active_faults)),
        (NODEID_INSTRUMENTS_NOT_GOOD, not_good),
    ]


def sensor_qualities(msg: pb.PlantStatusMsg) -> dict[int, int]:
    """Instrument quality per node fed by that instrument."""
    return {
        SENSOR_TO_NODEID[sensor.sensor_id]: int(sensor.quality)
        for sensor in msg.sensors
        if sensor.sensor_id in SENSOR_TO_NODEID
    }


def _control_mode(mode: int) -> str:
    """The mode's name; proto3 enums are open, and a newer PLC may send a value unknown here."""
    try:
        return str(pb.ControlMode.Name(mode)).lower()
    except ValueError:
        return "unknown"


def plc_updates(status: pb.PLCStatusMsg) -> list[Update]:
    trip = status.active_trip
    trip_cause = (
        f"{trip.parameter}={trip.value:g} (limit {trip.threshold:g})"
        if status.emergency_stop_active and trip.parameter
        else ""
    )
    return [
        (NODEID_PLC_MODE, _control_mode(status.mode)),
        (NODEID_EMERGENCY_STOP_ACTIVE, status.emergency_stop_active),
        (NODEID_TRIP_CAUSE, trip_cause),
        (NODEID_RESET_PERMITTED, status.reset_permitted),
        (NODEID_RESET_BLOCKERS, list(status.reset_blockers)),
        (NODEID_LOAD_DEMAND, status.load_demand_w),
        (NODEID_LOAD_SETPOINT, status.load_setpoint_w),
        (NODEID_PRESSURE_SETPOINT, status.setpoints.pressure_pa),
        (NODEID_LEVEL_SETPOINT, status.setpoints.water_level_m),
        (NODEID_STEAM_TEMP_SETPOINT, status.setpoints.steam_temp_k),
        (NODEID_WARNING_COUNT, int(status.warning_count)),
        (NODEID_TRIP_COUNT, int(status.trip_count)),
        (NODEID_PLC_COMMUNICATION, True),
    ]


def alarm_updates(alarms: pb.AlarmListMsg) -> list[Update]:
    open_alarms = list(alarms.alarms)
    unacknowledged = sum(
        1
        for alarm in open_alarms
        if alarm.state
        in (pb.AlarmState.ALARM_ACTIVE_UNACK, pb.AlarmState.ALARM_CLEARED_UNACK)
    )
    critical_active = sum(
        1
        for alarm in open_alarms
        if alarm.severity == "critical"
        and alarm.state
        in (pb.AlarmState.ALARM_ACTIVE_UNACK, pb.AlarmState.ALARM_ACTIVE_ACK)
    )
    return [
        # The list is one page (client.OPEN_ALARMS_LISTED); total counts them all.
        (NODEID_OPEN_ALARM_COUNT, max(int(alarms.total), len(open_alarms))),
        (NODEID_UNACKNOWLEDGED_COUNT, unacknowledged),
        (NODEID_CRITICAL_ACTIVE_COUNT, critical_active),
        (
            NODEID_OPEN_ALARMS,
            [
                f"{alarm.alarm_id} | {alarm.severity} | "
                f"{_ALARM_STATES.get(alarm.state, 'UNKNOWN')} | {alarm.message}"
                for alarm in open_alarms
            ],
        ),
        (NODEID_ALARM_COMMUNICATION, True),
    ]
