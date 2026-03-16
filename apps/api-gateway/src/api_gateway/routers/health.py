"""
Health check endpoint.

GET /health — used by Kubernetes liveness and readiness probes.
No authentication required — the probe runs before any token exists.
"""

from __future__ import annotations

from fastapi import APIRouter

from api_gateway.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")  # type: ignore[misc]
async def health_check() -> dict[str, str]:
    """
    Liveness probe endpoint.

    Returns 200 OK as long as the process is running and the event
    loop is responsive. Does NOT check database or MQTT connectivity —
    those are covered by readiness probes (future work).
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }
