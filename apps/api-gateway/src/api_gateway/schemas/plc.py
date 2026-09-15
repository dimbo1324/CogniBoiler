"""Schemas for the PLC status endpoint. SI units throughout."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SetpointValues(BaseModel):
    pressure_pa: float
    water_level_m: float
    steam_temp_k: float


class ValveCommandValues(BaseModel):
    fuel_valve: float
    feedwater_valve: float
    steam_valve: float
    spray_valve: float
    source: str = Field(..., description="operator | pid | safety | scheduler")
    operator_id: str
    timestamp_ms: int


class TripCause(BaseModel):
    parameter: str
    value: float
    threshold: float
    timestamp_ms: int


class AlarmConditionResponse(BaseModel):
    key: str
    parameter: str
    severity: str
    direction: str
    value: float
    threshold: float
    message: str
    since_ms: int


class ControlLoopResponse(BaseModel):
    name: str
    setpoint: float
    measurement: float
    output: float
    unit: str


class PLCStatusResponse(BaseModel):
    """Mode, targets, working setpoints, loops and alarm conditions of the PLC."""

    mode: Literal["auto", "manual", "estop"]
    emergency_stop_active: bool
    trip_cause: TripCause | None
    reset_permitted: bool
    reset_blockers: list[str]
    load_demand_w: float
    load_setpoint_w: float
    setpoints: SetpointValues = Field(..., description="Targets set by an engineer.")
    active_setpoints: SetpointValues = Field(
        ..., description="Working setpoints ramping toward the targets."
    )
    latest_command: ValveCommandValues
    active_conditions: list[AlarmConditionResponse]
    loops: list[ControlLoopResponse]
    warning_count: int
    trip_count: int
    run_id: int
