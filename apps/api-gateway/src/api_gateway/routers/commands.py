"""Operator command endpoints backed by the live PLC service."""

from __future__ import annotations

from typing import Annotated

import cogniboiler_pb2 as pb2
import grpc
from fastapi import APIRouter, Depends, HTTPException, Request, status

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.clients import PLCGatewayClient
from api_gateway.schemas.command import (
    CommandAckResponse,
    ResetRequest,
    SetpointRequest,
    ValveCommandRequest,
)

router = APIRouter(prefix="/api/v1/commands", tags=["commands"])


def _plc_client(request: Request) -> PLCGatewayClient:
    """Resolve the shared PLC client from application state."""
    return request.app.state.plc_client  # type: ignore[no-any-return]


@router.post("/valve", response_model=CommandAckResponse)
async def send_valve_command(
    request: Request,
    body: ValveCommandRequest,
    token: Annotated[TokenData, Depends(require_role("operator"))],
) -> CommandAckResponse:
    """Send a live manual valve command to the PLC controller."""
    try:
        ack = await _plc_client(request).send_command(
            pb2.ControlCommandMsg(
                fuel_valve=body.fuel_valve,
                feedwater_valve=body.feedwater_valve,
                steam_valve=body.steam_valve,
                source=pb2.CommandSource.OPERATOR,
                operator_id=str(token["sub"]),
            )
        )
    except grpc.RpcError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PLCService unavailable: {exc}",
        ) from exc

    return CommandAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
    )


@router.post("/setpoint", response_model=CommandAckResponse)
async def update_setpoints(
    request: Request,
    body: SetpointRequest,
    token: Annotated[TokenData, Depends(require_role("engineer"))],
) -> CommandAckResponse:
    """Update live PLC setpoints and return the acceptance status."""
    try:
        ack = await _plc_client(request).update_setpoints(
            pb2.SetpointsMsg(
                pressure_pa=body.pressure_pa,
                water_level_m=body.water_level_m,
                steam_temp_k=body.steam_temp_k,
                timestamp_ms=0,
            )
        )
    except grpc.RpcError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PLCService unavailable: {exc}",
        ) from exc

    return CommandAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
    )


@router.post("/reset", response_model=CommandAckResponse)
async def reset_emergency_stop(
    request: Request,
    body: ResetRequest,
    token: Annotated[TokenData, Depends(require_role("engineer"))],
) -> CommandAckResponse:
    """Reset the PLC emergency stop latch and return to AUTO mode."""
    operator_id = body.operator_id or str(token["sub"])
    try:
        ack = await _plc_client(request).reset_emergency_stop(operator_id)
    except grpc.RpcError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PLCService unavailable: {exc}",
        ) from exc

    return CommandAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
    )
