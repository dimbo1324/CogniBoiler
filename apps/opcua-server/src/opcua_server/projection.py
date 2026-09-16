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


def _fields(message: Any, mapping: Mapping[str, int]) -> list[Update]:
    return [(node_id, getattr(message, field)) for field, node_id in mapping.items()]


def plant_updates(msg: pb.PlantStatusMsg) -> list[Update]:
    simulation = msg.simulation
    not_good = sum(
        1 for sensor in msg.sensors if sensor.quality != pb.SensorQuality.GOOD
    )
    return [
        *_fields(msg.actuators, ACTUATOR_FIELD_TO_NODEID),
        *_fields(msg.emissions, EMISSIONS_FIELD_TO_NODEID),
        *_fields(msg.condenser, CONDENSER_FIELD_TO_NODEID),
        *_fields(msg.performance, PERFORMANCE_FIELD_TO_NODEID),
        *_fields(msg.health, HEALTH_FIELD_TO_NODEID),
        (2600, simulation.scenario),
        (2601, int(simulation.run_id)),
        (2602, simulation.simulation_time_s),
        (2603, simulation.speed_factor),
        (2604, simulation.run_state == pb.SimulationRunState.SIMULATION_PAUSED),
        (2605, sorted(fault.label for fault in msg.active_faults)),
        (2606, not_good),
    ]


def sensor_qualities(msg: pb.PlantStatusMsg) -> dict[int, int]:
    """Instrument quality per node fed by that instrument."""
    return {
        SENSOR_TO_NODEID[sensor.sensor_id]: int(sensor.quality)
        for sensor in msg.sensors
        if sensor.sensor_id in SENSOR_TO_NODEID
    }


def plc_updates(status: pb.PLCStatusMsg) -> list[Update]:
    trip = status.active_trip
    trip_cause = (
        f"{trip.parameter}={trip.value:g} (limit {trip.threshold:g})"
        if status.emergency_stop_active and trip.parameter
        else ""
    )
    return [
        (2700, pb.ControlMode.Name(status.mode).lower()),
        (2701, status.emergency_stop_active),
        (2702, trip_cause),
        (2703, status.reset_permitted),
        (2704, list(status.reset_blockers)),
        (2705, status.load_demand_w),
        (2706, status.load_setpoint_w),
        (2707, status.setpoints.pressure_pa),
        (2708, status.setpoints.water_level_m),
        (2709, status.setpoints.steam_temp_k),
        (2710, int(status.warning_count)),
        (2711, int(status.trip_count)),
        (2712, True),
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
        (2800, len(open_alarms)),
        (2801, unacknowledged),
        (2802, critical_active),
        (
            2803,
            [
                f"{alarm.alarm_id} | {alarm.severity} | "
                f"{_ALARM_STATES.get(alarm.state, 'UNKNOWN')} | {alarm.message}"
                for alarm in open_alarms
            ],
        ),
        (2804, True),
    ]
