"""
Operator command endpoints backed by the live PLC service.

The PLC records the authenticated username as the operator of every command; the audit
row of the request carries the PLC's verdict as its outcome.
"""

from __future__ import annotations

import cogniboiler_pb2 as pb2
import grpc
from fastapi import APIRouter, Request

from api_gateway.audit import command_outcome, set_audit_detail, set_audit_outcome
from api_gateway.auth.rbac import EngineerUser, OperatorUser
from api_gateway.clients import PLCGatewayClient
from api_gateway.problems import upstream_unavailable
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


def _ack(request: Request, ack: pb2.CommandAck) -> CommandAckResponse:
    set_audit_outcome(request, command_outcome(ack.accepted, ack.reason))
    return CommandAckResponse(
        accepted=ack.accepted,
        reason=ack.reason,
        timestamp_ms=ack.timestamp_ms,
    )


@router.post("/valve", response_model=CommandAckResponse)
async def send_valve_command(
    request: Request,
    body: ValveCommandRequest,
    user: OperatorUser,
) -> CommandAckResponse:
    """Send a manual valve command to the PLC; the PLC switches to MANUAL."""
    command = pb2.ControlCommandMsg(
        fuel_valve=body.fuel_valve,
        feedwater_valve=body.feedwater_valve,
        steam_valve=body.steam_valve,
        source=pb2.CommandSource.OPERATOR,
        operator_id=user.username,
    )
    if body.spray_valve is not None:
        command.spray_valve = body.spray_valve
    try:
        ack = await _plc_client(request).send_command(command)
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return _ack(request, ack)


@router.post("/load", response_model=CommandAckResponse)
async def set_load_demand(
    request: Request,
    body: LoadDemandRequest,
    user: OperatorUser,
) -> CommandAckResponse:
    """Set the electrical load the PLC drives the unit to in AUTO."""
    try:
        ack = await _plc_client(request).set_load_demand(body.load_w, user.username)
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return _ack(request, ack)


@router.post("/mode", response_model=CommandAckResponse)
async def set_control_mode(
    request: Request,
    body: ControlModeRequest,
    user: OperatorUser,
) -> CommandAckResponse:
    """Switch the PLC between AUTO and MANUAL, or trip the unit."""
    try:
        ack = await _plc_client(request).set_control_mode(
            _MODES[body.mode], user.username
        )
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return _ack(request, ack)


@router.post("/setpoint", response_model=CommandAckResponse)
async def update_setpoints(
    request: Request,
    body: SetpointRequest,
    user: EngineerUser,
) -> CommandAckResponse:
    """Update live PLC setpoints and return the acceptance status."""
    try:
        ack = await _plc_client(request).update_setpoints(
            pb2.SetpointsMsg(
                pressure_pa=body.pressure_pa,
                water_level_m=body.water_level_m,
                steam_temp_k=body.steam_temp_k,
                timestamp_ms=0,
                operator_id=user.username,
            )
        )
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return _ack(request, ack)


@router.post("/reset", response_model=CommandAckResponse)
async def reset_emergency_stop(
    request: Request,
    user: EngineerUser,
    body: ResetRequest | None = None,
) -> CommandAckResponse:
    """
    Reset the PLC emergency stop latch once its cause is gone; back to AUTO.

    The reset is recorded under the authenticated user. An operator_id in the body no
    longer overrides that; it is kept in the audit detail as a note.
    """
    if body is not None and body.operator_id and body.operator_id != user.username:
        set_audit_detail(request, f"stated operator_id={body.operator_id}")
    try:
        ack = await _plc_client(request).reset_emergency_stop(user.username)
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return _ack(request, ack)
