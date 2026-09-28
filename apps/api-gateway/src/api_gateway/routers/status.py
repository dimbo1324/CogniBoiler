"""System status endpoint backed by the live PhysicsService."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api_gateway.auth.rbac import ViewerUser
from api_gateway.plant_state import quality_name
from api_gateway.problems import UPSTREAM_RESPONSES, upstream_call
from api_gateway.schemas.sensor import (
    BoilerStatusResponse,
    SystemStatusResponse,
    TurbineStatusResponse,
)
from api_gateway.upstreams import PHYSICS_SERVICE, PhysicsClient

router = APIRouter(prefix="/api/v1", tags=["status"], responses=UPSTREAM_RESPONSES)


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    request: Request,
    _: ViewerUser,
    physics: PhysicsClient,
) -> SystemStatusResponse:
    """Return the current boiler and turbine state from live gRPC."""
    async with upstream_call(request, PHYSICS_SERVICE):
        current = await physics.get_system_state()

    boiler = BoilerStatusResponse(
        pressure_pa=current.boiler.pressure_pa,
        water_level_m=current.boiler.water_level_m,
        water_temp_k=current.boiler.water_temp_k,
        flue_gas_temp_k=current.boiler.flue_gas_temp_k,
        internal_energy_j=current.boiler.internal_energy_j,
        timestamp_ms=current.boiler.timestamp_ms,
        quality=quality_name(current.boiler.quality),
    )
    turbine = TurbineStatusResponse(
        electrical_power_w=current.turbine.electrical_power_w,
        shaft_power_w=current.turbine.shaft_power_w,
        enthalpy_in_j_kg=current.turbine.enthalpy_in_j_kg,
        enthalpy_out_j_kg=current.turbine.enthalpy_out_j_kg,
        exhaust_pressure_pa=current.turbine.exhaust_pressure_pa,
        steam_flow_kg_s=current.turbine.steam_flow_kg_s,
        timestamp_ms=current.turbine.timestamp_ms,
        steam_temp_in_k=current.turbine.steam_temp_in_k,
    )
    return SystemStatusResponse(boiler=boiler, turbine=turbine)
