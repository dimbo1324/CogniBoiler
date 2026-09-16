"""Mapping of PhysicsService messages to the gateway's plant schemas."""

from __future__ import annotations

from typing import Any

import cogniboiler_pb2 as pb2
from google.protobuf.message import Message

from api_gateway.schemas.plant import (
    FaultResponse,
    PlantStateResponse,
    ScenarioListResponse,
    ScenarioResponse,
    SimulationAckResponse,
    SimulationStatusResponse,
)

_FAULT_KIND_NAMES: dict[int, str] = {
    int(pb2.FaultKind.FAULT_BURNER_FOULING): "burner_fouling",
    int(pb2.FaultKind.FAULT_STEAM_LEAK): "steam_leak",
    int(pb2.FaultKind.FAULT_FEEDWATER_PUMP_FAILURE): "feedwater_pump_failure",
    int(pb2.FaultKind.FAULT_VALVE_STUCK): "valve_stuck",
    int(pb2.FaultKind.FAULT_SENSOR_DRIFT): "sensor_drift",
    int(pb2.FaultKind.FAULT_SENSOR_FAILURE): "sensor_failure",
}
FAULT_KINDS_BY_NAME: dict[str, int] = {
    name: kind for kind, name in _FAULT_KIND_NAMES.items()
}


def quality_name(value: int) -> str:
    return str(pb2.SensorQuality.Name(value)).lower()


def _scalars(message: Message) -> dict[str, Any]:
    """Every scalar field of a flat message, by its contract name."""
    return {
        field.name: getattr(message, field.name)
        for field in message.DESCRIPTOR.fields
        if field.type != field.TYPE_MESSAGE
    }


def simulation_status(message: pb2.SimulationStatusMsg) -> SimulationStatusResponse:
    values = _scalars(message)
    values["run_state"] = (
        "paused"
        if message.run_state == pb2.SimulationRunState.SIMULATION_PAUSED
        else "running"
    )
    return SimulationStatusResponse.model_validate(values)


def fault(message: pb2.FaultMsg) -> FaultResponse:
    values = _scalars(message)
    values["kind"] = _FAULT_KIND_NAMES.get(int(message.kind), "unspecified")
    return FaultResponse.model_validate(values)


def plant_state(message: pb2.SystemStateMsg) -> PlantStateResponse:
    boiler = _scalars(message.boiler)
    boiler["quality"] = quality_name(message.boiler.quality)
    return PlantStateResponse.model_validate(
        {
            "timestamp_ms": message.boiler.timestamp_ms,
            "simulation": simulation_status(message.simulation),
            "boiler": boiler,
            "turbine": _scalars(message.turbine),
            "actuators": _scalars(message.actuators),
            "emissions": _scalars(message.emissions),
            "condenser": _scalars(message.condenser),
            "health": _scalars(message.health),
            "faults": [fault(item) for item in message.active_faults],
            "sensors": [
                {
                    "sensor_id": item.sensor_id,
                    "quality": quality_name(item.quality),
                    "measured_value": item.measured_value,
                }
                for item in message.sensors
            ],
        }
    )


def scenario_list(message: pb2.ScenarioListMsg) -> ScenarioListResponse:
    return ScenarioListResponse(
        scenarios=[
            ScenarioResponse(
                name=item.name, title=item.title, description=item.description
            )
            for item in message.scenarios
        ],
        current=message.current,
    )


def simulation_ack(message: pb2.SimulationAck) -> SimulationAckResponse:
    return SimulationAckResponse(
        accepted=message.accepted,
        reason=message.reason,
        timestamp_ms=message.timestamp_ms,
        status=simulation_status(message.status),
    )
