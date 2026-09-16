"""
Readiness of the gateway and of everything it depends on.

The database is required: without it nobody can be authenticated, so the gateway is not
ready. The PLC, physics, alarm and historian services are checked too; when one of them
is down the gateway still serves the rest and reports itself degraded.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text

from api_gateway.config import settings
from api_gateway.dependencies import session_scope

logger = logging.getLogger(__name__)

ComponentState = Literal["up", "down"]


class ComponentHealth(BaseModel):
    name: str
    state: ComponentState
    required: bool
    latency_ms: float | None = Field(..., description="Round trip of the check.")


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    components: list[ComponentHealth]
    checked_at_ms: int


async def _probe(
    name: str, required: bool, check: Callable[[], Awaitable[object]]
) -> ComponentHealth:
    started = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout=settings.ready_check_timeout_s)
    except Exception as exc:
        logger.debug("Readiness check %s failed: %s", name, exc)
        return ComponentHealth(
            name=name, state="down", required=required, latency_ms=None
        )
    return ComponentHealth(
        name=name,
        state="up",
        required=required,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


async def check_readiness(app: FastAPI) -> ReadinessResponse:
    async def database() -> None:
        async with session_scope(app) as session:
            await session.execute(text("SELECT 1"))

    async def historian() -> None:
        if not await asyncio.to_thread(app.state.historian_client.ping):
            raise ConnectionError("InfluxDB ping failed")

    components = await asyncio.gather(
        _probe("database", True, database),
        _probe("physics-engine", False, app.state.physics_client.health),
        _probe("plc-controller", False, app.state.plc_client.health),
        _probe("alert-manager", False, app.state.alarm_client.health),
        _probe("historian", False, historian),
    )
    if any(c.required and c.state == "down" for c in components):
        status: Literal["ready", "degraded", "not_ready"] = "not_ready"
    elif any(c.state == "down" for c in components):
        status = "degraded"
    else:
        status = "ready"
    return ReadinessResponse(
        status=status,
        components=list(components),
        checked_at_ms=int(time.time() * 1000),
    )
