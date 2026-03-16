"""
Pydantic schemas for sensor status and system state endpoints.
Values mirror the protobuf messages (BoilerStateMsg, TurbineStateMsg)
but are expressed as plain Python types for JSON serialisation.
All physical quantities use SI units — same convention as the physics engine.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoilerStatusResponse(BaseModel):
    """
    Current boiler state as returned by GET /api/v1/status.
    All values in SI units, matching BoilerStateMsg fields exactly
    so the gateway can populate this schema directly from protobuf.
    """

    pressure_pa: float = Field(..., description="Drum pressure [Pa].")
    water_level_m: float = Field(..., description="Water level in drum [m].")
    water_temp_k: float = Field(..., description="Bulk water temperature [K].")
    flue_gas_temp_k: float = Field(..., description="Flue gas temperature [K].")
    internal_energy_j: float = Field(..., description="Total internal energy [J].")
    timestamp_ms: int = Field(..., description="UTC epoch milliseconds.")
    quality: str = Field(..., description="Sensor quality: good | uncertain | bad.")


class TurbineStatusResponse(BaseModel):
    """
    Current turbine state as returned by GET /api/v1/status.
    All values in SI units, matching TurbineStateMsg fields exactly.
    """

    electrical_power_w: float = Field(..., description="Net electrical output [W].")
    shaft_power_w: float = Field(..., description="Mechanical shaft power [W].")
    enthalpy_in_j_kg: float = Field(..., description="Inlet specific enthalpy [J/kg].")
    enthalpy_out_j_kg: float = Field(
        ..., description="Outlet specific enthalpy [J/kg]."
    )
    exhaust_pressure_pa: float = Field(..., description="Condenser back-pressure [Pa].")
    steam_flow_kg_s: float = Field(..., description="Steam mass flow [kg/s].")
    timestamp_ms: int = Field(..., description="UTC epoch milliseconds.")


class SystemStatusResponse(BaseModel):
    """
    Full system state: boiler + turbine combined.
    Returned by GET /api/v1/status.
    """

    boiler: BoilerStatusResponse
    turbine: TurbineStatusResponse
