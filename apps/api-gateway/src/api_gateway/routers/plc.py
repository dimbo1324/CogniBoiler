"""PLC status endpoint backed by the live PLCService."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api_gateway.auth.rbac import ViewerUser
from api_gateway.plc_state import plc_status_from_proto
from api_gateway.problems import UPSTREAM_RESPONSES, upstream_call
from api_gateway.schemas.plc import PLCStatusResponse
from api_gateway.upstreams import PLC_SERVICE, PlcClient

router = APIRouter(prefix="/api/v1/plc", tags=["plc"], responses=UPSTREAM_RESPONSES)


@router.get("/status", response_model=PLCStatusResponse)
async def get_plc_status(
    request: Request,
    _: ViewerUser,
    plc: PlcClient,
) -> PLCStatusResponse:
    """Current PLC mode, targets, loops, alarm conditions and reset permission."""
    async with upstream_call(request, PLC_SERVICE):
        message = await plc.get_control_status()
    return plc_status_from_proto(message)
