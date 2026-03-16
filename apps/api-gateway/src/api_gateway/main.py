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

import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.config import settings
from api_gateway.routers import auth, commands, health, status, websocket

logger = logging.getLogger(__name__)

# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Code before `yield` runs on startup.
    Code after  `yield` runs on shutdown.

    Phase 5.4 will add:
      - SQLAlchemy async engine initialisation
      - MQTT client connection
      - gRPC channel pool warm-up
    """
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    yield
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
    @app.middleware("http")  # type: ignore[misc]
    async def add_process_time_header(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(status.router)
    app.include_router(commands.router)
    app.include_router(websocket.router)

    return app


# ─── Application instance ─────────────────────────────────────────────────────

app = create_app()
