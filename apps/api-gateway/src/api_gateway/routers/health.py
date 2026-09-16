"""
Liveness, readiness and platform state.

GET /health — the process is alive (container healthcheck). No dependencies checked.
GET /ready  — the gateway can serve: 200 when ready or degraded, 503 without a database.
GET /api/v1/platform — readiness plus live data flow, for the console's platform screen.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from api_gateway.auth.rbac import ViewerUser
from api_gateway.config import settings
from api_gateway.readiness import ReadinessResponse, check_readiness
from api_gateway.realtime.hub import RealtimeHub

router = APIRouter(tags=["health"])


class PlatformResponse(BaseModel):
    readiness: ReadinessResponse
    telemetry_age_s: float | None = Field(
        ..., description="Seconds since the gateway last received a plant state."
    )
    websocket_clients: int


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe: the process runs and its event loop answers."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe with the state of the database and every upstream service."""
    result = await check_readiness(request.app)
    if result.status == "not_ready":
        response.status_code = 503
    return result


@router.get("/api/v1/platform", response_model=PlatformResponse, tags=["platform"])
async def platform(request: Request, _: ViewerUser) -> PlatformResponse:
    hub: RealtimeHub | None = getattr(request.app.state, "realtime_hub", None)
    return PlatformResponse(
        readiness=await check_readiness(request.app),
        telemetry_age_s=hub.telemetry_age_s() if hub is not None else None,
        websocket_clients=hub.subscriber_count if hub is not None else 0,
    )
