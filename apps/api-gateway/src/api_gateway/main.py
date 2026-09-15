"""
CogniBoiler API Gateway — FastAPI application entry point.

This module creates the FastAPI application instance, registers all
routers, and configures middleware. It is the single file that ties
the entire gateway together.

Starting the server:
    uvicorn api_gateway.main:app --reload --port 8000

In production (Phase 7):
    uvicorn api_gateway.main:app \
        --host 0.0.0.0 --port 8000 \
        --ssl-keyfile certs/server.key \
        --ssl-certfile certs/server.crt \
        --workers 4
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.auth.jwt_handler import decode_access_token
from api_gateway.clients import (
    AlarmGatewayClient,
    AlarmGatewayConfig,
    HistorianQueryClient,
    HistorianQueryConfig,
    PhysicsGatewayClient,
    PhysicsGatewayConfig,
    PLCGatewayClient,
    PLCGatewayConfig,
)
from api_gateway.config import settings
from api_gateway.db_init import ensure_schema_and_seed_defaults
from api_gateway.dependencies import AsyncSessionLocal
from api_gateway.models.user import AuditLog
from api_gateway.routers import (
    alarms,
    audit,
    auth,
    commands,
    health,
    history,
    plc,
    status,
    websocket,
)

logger = logging.getLogger(__name__)

# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Application lifespan manager.

    Code before `yield` runs on startup.
    Code after  `yield` runs on shutdown.

    Phase 5.4 will add:
      - SQLAlchemy async engine initialisation
      - MQTT client connection
      - gRPC channel pool warm-up
    """
    if settings.auto_init_db:
        await ensure_schema_and_seed_defaults()

    app.state.physics_client = PhysicsGatewayClient(
        PhysicsGatewayConfig(target=settings.physics_grpc_target)
    )
    app.state.plc_client = PLCGatewayClient(
        PLCGatewayConfig(target=settings.plc_grpc_target)
    )
    app.state.alarm_client = AlarmGatewayClient(
        AlarmGatewayConfig(target=settings.alarm_grpc_target)
    )
    app.state.historian_client = HistorianQueryClient(
        HistorianQueryConfig(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org,
            bucket=settings.influx_bucket,
        )
    )

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    yield
    await app.state.physics_client.close()
    await app.state.plc_client.close()
    await app.state.alarm_client.close()
    app.state.historian_client.close()
    logger.info("Shutting down %s", settings.app_name)


# ─── Application factory ──────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separated from module-level instantiation so that tests can call
    create_app() to get a fresh instance with overridden dependencies,
    without importing side-effects at module level.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "REST API gateway for the CogniBoiler digital twin platform. "
            "Provides authenticated access to boiler/turbine state, "
            "operator commands, and real-time WebSocket streaming."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # In production replace ["*"] with the actual frontend origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    # Adds X-Process-Time header to every response.
    @app.middleware("http")
    async def add_process_time_header(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        request_body = await request.body()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"

        request_hash = (
            hashlib.sha256(request_body).hexdigest() if request_body else None
        )
        user_id: int | None = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.removeprefix("Bearer ").strip()
            try:
                payload = decode_access_token(token)
                user_id = int(str(payload["sub"]))
            except Exception:
                user_id = None

        try:
            async with AsyncSessionLocal() as session:
                session.add(
                    AuditLog(
                        user_id=user_id,
                        ip_address=request.client.host if request.client else "unknown",
                        method=request.method,
                        endpoint=request.url.path,
                        request_body_hash=request_hash,
                        response_status=response.status_code,
                        duration_ms=int(elapsed * 1000),
                        timestamp_ms=int(time.time() * 1000),
                        detail=request.url.query or None,
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("Audit write skipped: %s", exc)

        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(status.router)
    app.include_router(commands.router)
    app.include_router(plc.router)
    app.include_router(history.router)
    app.include_router(alarms.router)
    app.include_router(audit.router)
    app.include_router(websocket.router)

    return app


# ─── Application instance ─────────────────────────────────────────────────────

app = create_app()
