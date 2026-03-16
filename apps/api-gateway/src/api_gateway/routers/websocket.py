"""
WebSocket endpoint for real-time sensor data streaming.

WS /ws/realtime — authenticated clients receive boiler + turbine
state updates once per second.

Authentication over WebSocket:
    Standard HTTP Authorization headers are not available in browser
    WebSocket connections. Instead the client passes the access token
    as a query parameter: ws://host/ws/realtime?token=<jwt>

    This is acceptable because:
      1. The connection is over TLS (wss://) in production
      2. The token is short-lived (15 min)
      3. Query params appear in server access logs — warn users not
         to log these in production (TODO: move to first-message auth)

In production the gateway subscribes to MQTT sensors/# and forwards
messages to all connected WebSocket clients. For Phase 5.1 it sends
a synthetic heartbeat every second so the WebSocket layer can be
tested without a running MQTT broker.
"""

from __future__ import annotations

import asyncio
import json
import time

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api_gateway.auth.jwt_handler import decode_access_token

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/realtime")  # type: ignore[misc]
async def realtime_stream(websocket: WebSocket) -> None:
    """
    Stream real-time sensor data to authenticated WebSocket clients.

    Authentication:
        Pass the JWT access token as ?token=<jwt> query parameter.
        Connection is rejected with 4001 if the token is missing,
        expired, or invalid.

    Message format (JSON, sent every second):
        {
            "ts_ms": 1710000000000,
            "boiler": { "pressure_pa": 140000000, ... },
            "turbine": { "electrical_power_w": 200000000, ... }
        }
    """
    # ── Token authentication ──────────────────────────────────────────────────
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token.")
        return

    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as exc:
        await websocket.close(code=4001, reason=f"Invalid token: {exc}")
        return

    await websocket.accept()

    # ── Streaming loop ────────────────────────────────────────────────────────
    try:
        while True:
            # TODO (Phase 5.4): subscribe to MQTT sensors/# and forward
            # real messages instead of this synthetic stub.
            now_ms = int(time.time() * 1000)
            message = {
                "ts_ms": now_ms,
                "user": payload.get("sub"),
                "boiler": {
                    "pressure_pa": 140.0e5,
                    "water_level_m": 4.8,
                    "water_temp_k": 611.0,
                },
                "turbine": {
                    "electrical_power_w": 200.0e6,
                    "steam_flow_kg_s": 150.0,
                },
            }
            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        # Client closed the connection — clean exit, no error logging needed
        pass
