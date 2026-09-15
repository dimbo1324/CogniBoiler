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
    ControlModeRequest,
    LoadDemandRequest,
    ResetRequest,
    SetpointRequest,
    ValveCommandRequest,
)

router = APIRouter(prefix="/api/v1/commands", tags=["commands"])

_MODES: dict[str, int] = {
    "auto": int(pb2.ControlMode.AUTO),
    "manual": int(pb2.ControlMode.MANUAL),
    "estop": int(pb2.ControlMode.ESTOP),
}


def _plc_client(request: Request) -> PLCGatewayClient:
    """Resolve the shared PLC client from application state."""
    return request.app.state.plc_client  # type: ignore[no-any-return]


def _ack(ack: pb2.CommandAck) -> CommandAckResponse:
    return CommandAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
    )


def _unavailable(exc: grpc.RpcError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"PLCService unavailable: {exc}",
    )


@router.post("/valve", response_model=CommandAckResponse)
async def send_valve_command(
    request: Request,
    body: ValveCommandRequest,
    token: Annotated[TokenData, Depends(require_role("operator"))],
) -> CommandAckResponse:
    """Send a manual valve command to the PLC; the PLC switches to MANUAL."""
    command = pb2.ControlCommandMsg(
        fuel_valve=body.fuel_valve,
        feedwater_valve=body.feedwater_valve,
        steam_valve=body.steam_valve,
        source=pb2.CommandSource.OPERATOR,
        operator_id=str(token["sub"]),
    )
    if body.spray_valve is not None:
        command.spray_valve = body.spray_valve
    try:
        ack = await _plc_client(request).send_command(command)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _ack(ack)


@router.post("/load", response_model=CommandAckResponse)
async def set_load_demand(
    request: Request,
    body: LoadDemandRequest,
    token: Annotated[TokenData, Depends(require_role("operator"))],
) -> CommandAckResponse:
    """Set the electrical load the PLC drives the unit to in AUTO."""
    try:
        ack = await _plc_client(request).set_load_demand(body.load_w, str(token["sub"]))
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _ack(ack)


@router.post("/mode", response_model=CommandAckResponse)
async def set_control_mode(
    request: Request,
    body: ControlModeRequest,
    token: Annotated[TokenData, Depends(require_role("operator"))],
) -> CommandAckResponse:
    """Switch the PLC between AUTO and MANUAL, or trip the unit."""
    try:
        ack = await _plc_client(request).set_control_mode(
            _MODES[body.mode], str(token["sub"])
        )
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _ack(ack)


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
                operator_id=str(token["sub"]),
            )
        )
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _ack(ack)


@router.post("/reset", response_model=CommandAckResponse)
async def reset_emergency_stop(
    request: Request,
    body: ResetRequest,
    token: Annotated[TokenData, Depends(require_role("engineer"))],
) -> CommandAckResponse:
    """Reset the PLC emergency stop latch once its cause is gone; back to AUTO."""
    operator_id = body.operator_id or str(token["sub"])
    try:
        ack = await _plc_client(request).reset_emergency_stop(operator_id)
    except grpc.RpcError as exc:
        raise _unavailable(exc) from exc
    return _ack(ack)
