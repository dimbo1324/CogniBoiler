"""
Operator command endpoints.

POST /api/v1/commands/valve     — set valve positions (operator+)
POST /api/v1/commands/setpoint  — set PID setpoints  (engineer+)

In production these forward commands to the PLC Controller via gRPC.
For Phase 5.1 they validate input and return a stub acknowledgement.

Role requirements:
    valve    -> minimum: operator
    setpoint -> minimum: engineer
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends

from api_gateway.auth.jwt_handler import TokenData
from api_gateway.auth.rbac import require_role
from api_gateway.schemas.command import (
    CommandAckResponse,
    SetpointRequest,
    ValveCommandRequest,
)

router = APIRouter(prefix="/api/v1/commands", tags=["commands"])


@router.post("/valve", response_model=CommandAckResponse)  # type: ignore[misc]
async def send_valve_command(
    body: ValveCommandRequest,
    _: Annotated[TokenData, Depends(require_role("operator"))],
) -> CommandAckResponse:
    """
    Send valve position commands to the PLC controller.

    Requires: operator role or above.
    All three valve positions [0.0–1.0] must be provided together.

    TODO (Phase 5.4): forward to PLCService via gRPC.
    """
    return CommandAckResponse(
        accepted=True,
        reason="",
        timestamp_ms=int(time.time() * 1000),
    )


@router.post("/setpoint", response_model=CommandAckResponse)  # type: ignore[misc]
async def update_setpoints(
    body: SetpointRequest,
    _: Annotated[TokenData, Depends(require_role("engineer"))],
) -> CommandAckResponse:
    """
    Update PID controller setpoints.

    Requires: engineer role or above.
    Setpoint values are validated against safe operating ranges by
    Pydantic before this function is called (see SetpointRequest).

    TODO (Phase 5.4): forward to PLCService via gRPC.
    """
    return CommandAckResponse(
        accepted=True,
        reason="",
        timestamp_ms=int(time.time() * 1000),
    )
