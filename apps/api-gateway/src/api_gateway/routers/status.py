"""
System status endpoint.

GET /api/v1/status — returns current boiler + turbine state.

In production this will query the Physics Engine via gRPC
(PhysicsService.GetSystemState). For Phase 5.1 it returns a
realistic stub so the routing, auth, and schema layers can be
tested without a running physics engine.

Minimum role required: viewer (any authenticated user).
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.schemas.sensor import (
    BoilerStatusResponse,
    SystemStatusResponse,
    TurbineStatusResponse,
)

router = APIRouter(prefix="/api/v1", tags=["status"])


@router.get("/status", response_model=SystemStatusResponse)  # type: ignore[misc]
async def get_system_status(
    _: Annotated[TokenData, Depends(require_role("viewer"))],
) -> SystemStatusResponse:
    """
    Return the current boiler and turbine state.

    Requires: viewer role or above.

    TODO (Phase 5.4): replace stub with gRPC call to PhysicsService:
        channel = grpc.aio.insecure_channel("physics-engine:50051")
        stub = PhysicsServiceStub(channel)
        response = await stub.GetSystemState(Empty())
    """
    now_ms = int(time.time() * 1000)

    # Stub values — nominal operating point of the 300 MW boiler
    boiler = BoilerStatusResponse(
        pressure_pa=140.0e5,
        water_level_m=4.8,
        water_temp_k=611.0,
        flue_gas_temp_k=1200.0,
        internal_energy_j=2.5e12,
        timestamp_ms=now_ms,
        quality="good",
    )
    turbine = TurbineStatusResponse(
        electrical_power_w=200.0e6,
        shaft_power_w=205.0e6,
        enthalpy_in_j_kg=3_400_000.0,
        enthalpy_out_j_kg=2_200_000.0,
        exhaust_pressure_pa=7_000.0,
        steam_flow_kg_s=150.0,
        timestamp_ms=now_ms,
    )
    return SystemStatusResponse(boiler=boiler, turbine=turbine)
