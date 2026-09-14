"""System status endpoint backed by the live PhysicsService."""

from __future__ import annotations

from typing import Annotated

import cogniboiler_pb2 as pb2
import grpc
from fastapi import APIRouter, Depends, HTTPException, Request, status

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.clients import PhysicsGatewayClient
from api_gateway.schemas.sensor import (
    BoilerStatusResponse,
    SystemStatusResponse,
    TurbineStatusResponse,
)

router = APIRouter(prefix="/api/v1", tags=["status"])


def _physics_client(request: Request) -> PhysicsGatewayClient:
    """Resolve the shared PhysicsService client from application state."""
    return request.app.state.physics_client  # type: ignore[no-any-return]


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    request: Request,
    _: Annotated[TokenData, Depends(require_role("viewer"))],
) -> SystemStatusResponse:
    """Return the current boiler and turbine state from live gRPC."""
    try:
        current = await _physics_client(request).get_system_state()
    except grpc.RpcError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PhysicsService unavailable: {exc}",
        ) from exc

    boiler = BoilerStatusResponse(
        pressure_pa=current.boiler.pressure_pa,
        water_level_m=current.boiler.water_level_m,
        water_temp_k=current.boiler.water_temp_k,
        flue_gas_temp_k=current.boiler.flue_gas_temp_k,
        internal_energy_j=current.boiler.internal_energy_j,
        timestamp_ms=current.boiler.timestamp_ms,
        quality=pb2.SensorQuality.Name(current.boiler.quality).lower(),
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
