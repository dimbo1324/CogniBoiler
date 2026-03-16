"""
Pydantic schemas for operator command endpoints.
All valve positions are normalised to [0.0, 1.0] — same convention
as ControlCommandMsg in the protobuf schema.
Setpoint values use SI units throughout.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValveCommandRequest(BaseModel):
    """
    Request body for POST /api/v1/commands/valve.
    Requires minimum role: operator.
    All three valve positions must be provided together to avoid
    partial state updates that could destabilise the control loop.
    """

    fuel_valve: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fuel valve position [0.0 = closed, 1.0 = fully open].",
        examples=[0.75],
    )
    feedwater_valve: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Feedwater valve position [0.0 – 1.0].",
        examples=[0.6],
    )
    steam_valve: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Steam bypass valve position [0.0 – 1.0].",
        examples=[0.5],
    )


class SetpointRequest(BaseModel):
    """
    Request body for POST /api/v1/commands/setpoint.
    Requires minimum role: engineer.
    All three setpoints must be provided together — the cascade PID
    controller uses them as a consistent set.
    """

    pressure_pa: float = Field(
        ...,
        ge=50e5,
        le=185e5,
        description="Target drum pressure [Pa]. Safe range: 50–185 bar.",
        examples=[140e5],
    )
    water_level_m: float = Field(
        ...,
        ge=0.5,
        le=9.0,
        description="Target water level in drum [m].",
        examples=[4.8],
    )
    steam_temp_k: float = Field(
        ...,
        ge=400.0,
        le=848.0,
        description="Target steam temperature [K]. Safe range: 400–848 K.",
        examples=[811.0],
    )


class CommandAckResponse(BaseModel):
    """
    Response body for all command endpoints.
    Mirrors CommandAck from the protobuf schema.
    If accepted=False, reason explains why the command was rejected
    (e.g. value out of safe range, safety interlock active).
    """

    accepted: bool = Field(..., description="True if the command was accepted.")
    reason: str = Field(
        default="",
        description="Rejection reason. Empty string when accepted=True.",
    )
    timestamp_ms: int = Field(..., description="UTC epoch milliseconds.")
