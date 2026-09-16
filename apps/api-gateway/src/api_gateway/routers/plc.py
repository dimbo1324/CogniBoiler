"""PLC status endpoint backed by the live PLCService."""

from __future__ import annotations

import grpc
from fastapi import APIRouter, Request

from api_gateway.auth.rbac import ViewerUser
from api_gateway.clients import PLCGatewayClient
from api_gateway.plc_state import plc_status_from_proto
from api_gateway.problems import upstream_unavailable
from api_gateway.schemas.plc import PLCStatusResponse

router = APIRouter(prefix="/api/v1/plc", tags=["plc"])


def _plc_client(request: Request) -> PLCGatewayClient:
    """Resolve the shared PLC client from application state."""
    return request.app.state.plc_client  # type: ignore[no-any-return]


@router.get("/status", response_model=PLCStatusResponse)
async def get_plc_status(
    request: Request,
    _: ViewerUser,
) -> PLCStatusResponse:
    """Current PLC mode, targets, loops, alarm conditions and reset permission."""
    try:
        message = await _plc_client(request).get_control_status()
    except grpc.RpcError as exc:
        raise upstream_unavailable("PLCService", exc) from exc
    return plc_status_from_proto(message)
