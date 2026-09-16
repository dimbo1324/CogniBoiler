"""Mapping of the PLC's protobuf status to the REST and WebSocket schema."""

from __future__ import annotations

from typing import Literal, cast

import cogniboiler_pb2 as pb2

from api_gateway.schemas.plc import (
    AlarmConditionResponse,
    ControlLoopResponse,
    PLCStatusResponse,
    SetpointValues,
    TripCause,
    ValveCommandValues,
)


def _setpoints(message: pb2.SetpointsMsg) -> SetpointValues:
    return SetpointValues(
        pressure_pa=message.pressure_pa,
        water_level_m=message.water_level_m,
        steam_temp_k=message.steam_temp_k,
    )


def plc_status_from_proto(message: pb2.PLCStatusMsg) -> PLCStatusResponse:
    """Map the PLC's protobuf status to the REST schema."""
    command = message.latest_command
    trip = message.active_trip
    mode = cast(
        Literal["auto", "manual", "estop"], pb2.ControlMode.Name(message.mode).lower()
    )
    return PLCStatusResponse(
        mode=mode,
        emergency_stop_active=message.emergency_stop_active,
        trip_cause=(
            TripCause(
                parameter=trip.parameter,
                value=trip.value,
                threshold=trip.threshold,
                timestamp_ms=trip.timestamp_ms,
            )
            if message.emergency_stop_active and trip.parameter
            else None
        ),
        reset_permitted=message.reset_permitted,
        reset_blockers=list(message.reset_blockers),
        load_demand_w=message.load_demand_w,
        load_setpoint_w=message.load_setpoint_w,
        setpoints=_setpoints(message.setpoints),
        active_setpoints=_setpoints(message.active_setpoints),
        latest_command=ValveCommandValues(
            fuel_valve=command.fuel_valve,
            feedwater_valve=command.feedwater_valve,
            steam_valve=command.steam_valve,
            spray_valve=command.spray_valve,
            source=pb2.CommandSource.Name(command.source).lower(),
            operator_id=command.operator_id,
            timestamp_ms=command.timestamp_ms,
        ),
        active_conditions=[
            AlarmConditionResponse(
                key=condition.key,
                parameter=condition.parameter,
                severity=condition.severity,
                direction=condition.direction,
                value=condition.value,
                threshold=condition.threshold,
                message=condition.message,
                since_ms=condition.since_ms,
            )
            for condition in message.active_conditions
        ],
        loops=[
            ControlLoopResponse(
                name=loop.name,
                setpoint=loop.setpoint,
                measurement=loop.measurement,
                output=loop.output,
                unit=loop.unit,
            )
            for loop in message.loops
        ],
        warning_count=message.warning_count,
        trip_count=message.trip_count,
        run_id=message.run_id,
    )
