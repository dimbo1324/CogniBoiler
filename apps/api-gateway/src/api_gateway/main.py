"""
CogniBoiler API Gateway — FastAPI application entry point.

Starting the server:
    uvicorn api_gateway.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.audit import AuditMiddleware
from api_gateway.auth.throttle import LoginThrottle, ThrottlePolicy
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
from api_gateway.db_init import seed_roles_and_demo_users
from api_gateway.observability import ObservabilityMiddleware, observe_app
from api_gateway.observability import router as metrics_router
from api_gateway.problems import install_problem_handlers
from api_gateway.realtime.hub import RealtimeHub
from api_gateway.realtime.sources import run_mqtt_events, run_plc_status, run_telemetry
from api_gateway.routers import (
    alarms,
    audit,
    auth,
    commands,
    health,
    history,
    kpi,
    plc,
    simulation,
    status,
    users,
    websocket,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Upstream clients and realtime sources live exactly as long as the application."""
    if settings.auto_init_db:
        await seed_roles_and_demo_users()

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
            aggregate_bucket=settings.influx_aggregate_bucket,
            raw_retention_days=settings.influx_raw_retention_days,
        )
    )
    hub = RealtimeHub(
        queue_size=settings.ws_send_queue_size, max_rate_hz=settings.ws_max_rate_hz
    )
    app.state.realtime_hub = hub
    sources = [
        asyncio.create_task(
            run_telemetry(hub, app.state.physics_client), name="realtime-telemetry"
        ),
        asyncio.create_task(
            run_plc_status(
                hub, app.state.plc_client, settings.ws_plc_status_interval_s
            ),
            name="realtime-plc-status",
        ),
        asyncio.create_task(
            run_mqtt_events(
                hub,
                settings.mqtt_host,
                settings.mqtt_port,
                settings.mqtt_username or None,
                settings.mqtt_password or None,
            ),
            name="realtime-mqtt-events",
        ),
    ]

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    try:
        yield
    finally:
        for task in sources:
            task.cancel()
        await asyncio.gather(*sources, return_exceptions=True)
        await app.state.physics_client.close()
        await app.state.plc_client.close()
        await app.state.alarm_client.close()
        app.state.historian_client.close()
        logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separated from module-level instantiation so that tests can call
    create_app() to get a fresh instance with overridden dependencies.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "REST and WebSocket gateway of the CogniBoiler digital twin: sign-in and "
            "sessions, plant state, PLC commands, simulation control, alarms, history, "
            "audit and user administration. Errors are Problem Details (RFC 9457)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.login_throttle = LoginThrottle(
        per_account=ThrottlePolicy(
            settings.login_max_failures_per_account, settings.login_failure_window_s
        ),
        per_client=ThrottlePolicy(
            settings.login_max_failures_per_client, settings.login_failure_window_s
        ),
    )
    install_problem_handlers(app)
    observe_app(app)

    app.add_middleware(AuditMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After", "X-Process-Time", "X-Correlation-ID"],
    )
    # Added last, so it runs first: the audit entry and every log line of a request
    # already carry its correlation id.
    app.add_middleware(ObservabilityMiddleware)

    for router in (
        health.router,
        auth.router,
        status.router,
        simulation.router,
        commands.router,
        plc.router,
        history.router,
        kpi.router,
        alarms.router,
        audit.router,
        users.router,
        websocket.router,
        metrics_router,
    ):
        app.include_router(router)
    return app


app = create_app()
