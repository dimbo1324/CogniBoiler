"""
Correlation ids and HTTP metrics at the gateway's edge.

Every HTTP request and WebSocket connection runs under a correlation id: the caller's
`X-Correlation-ID` when it has a safe shape, otherwise a new one. The id is returned in the
response header, stamped on every log line written while the request runs, and passed on
in gRPC metadata to the PLC, physics and alarm services. Requests are counted and timed by
route template, so an id in a path never becomes a metric label.
"""

from __future__ import annotations

import time

from cogniboiler_observability import CORRELATION_HEADER, correlation_scope
from fastapi import APIRouter, FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests answered.", ["method", "route", "status"]
)
HTTP_SECONDS = Histogram(
    "http_request_seconds",
    "Time to answer an HTTP request.",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
WEBSOCKET_CLIENTS = Gauge("gateway_websocket_clients", "Open WebSocket connections.")
TELEMETRY_AGE = Gauge(
    "gateway_telemetry_age_seconds",
    "Seconds since the gateway last received a plant state; -1 before the first.",
)

_HEADER = CORRELATION_HEADER.lower().encode("latin-1")

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition; scraped inside the Compose network."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _caller_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", ()):
        if name == _HEADER:
            return bytes(value).decode("latin-1")
    return None


def _route(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        with correlation_scope(_caller_id(scope)) as correlation_id:
            if scope["type"] == "websocket":
                await self.app(scope, receive, send)
                return

            started = time.perf_counter()
            status = 500

            async def send_with_id(message: Message) -> None:
                nonlocal status
                if message["type"] == "http.response.start":
                    status = int(message["status"])
                    headers = list(message.get("headers", []))
                    headers.append((_HEADER, correlation_id.encode("latin-1")))
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, send_with_id)
            finally:
                method = str(scope["method"])
                route = _route(scope)
                HTTP_REQUESTS.labels(method, route, str(status)).inc()
                HTTP_SECONDS.labels(method, route).observe(
                    time.perf_counter() - started
                )


def observe_app(app: FastAPI) -> None:
    """Read the live-channel gauges from the application's hub when Prometheus scrapes."""

    def clients() -> float:
        hub = getattr(app.state, "realtime_hub", None)
        return float(hub.subscriber_count) if hub is not None else 0.0

    def telemetry_age() -> float:
        hub = getattr(app.state, "realtime_hub", None)
        age = hub.telemetry_age_s() if hub is not None else None
        return float(age) if age is not None else -1.0

    WEBSOCKET_CLIENTS.set_function(clients)
    TELEMETRY_AGE.set_function(telemetry_age)
