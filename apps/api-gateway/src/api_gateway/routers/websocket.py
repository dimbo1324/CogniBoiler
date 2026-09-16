"""
WebSocket /ws: live channels telemetry, plc and alarms.

Protocol (JSON text frames):

  client → {"type": "auth", "access_token": "..."}          first frame, within 5 s
  server → {"type": "welcome", "user": ..., "role": ..., "channels": [...],
            "max_rate_hz": 10, "token_expires_at_ms": ...}
  client → {"type": "subscribe", "channels": ["telemetry", "alarms"], "max_rate_hz": 2}
  client → {"type": "unsubscribe", "channels": ["telemetry"]}
  client → {"type": "auth", "access_token": "..."}          a fresh token of the same user
  client → {"type": "ping"}                                  server → {"type": "pong"}
  server → {"type": "data", "channel": "plc", "kind": "status" | "event", "ts_ms": ...,
            "data": {...}}
  server → {"type": "error", "code": "...", "detail": "..."}

The token goes in the first frame, not the URL, so it never lands in access logs. The
connection is closed with 4401 when the token expires without being renewed, when the
session is closed or the account blocked (checked every 30 s), and with 1013 when the
client cannot keep up with PLC events or alarm changes. Telemetry is sent at most at the
requested rate, capped by the server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from api_gateway.audit import AuditActor, client_address, write_audit_entry
from api_gateway.auth.identity import (
    AuthenticationError,
    CurrentUser,
    resolve_access_token,
)
from api_gateway.config import settings
from api_gateway.dependencies import session_scope
from api_gateway.models.user import AuditLog
from api_gateway.realtime.hub import Channel, RealtimeHub, Subscriber, encode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

CLOSE_UNAUTHORIZED = 4401
CLOSE_BAD_REQUEST = 4400
CLOSE_TRY_AGAIN_LATER = 1013
REVALIDATE_INTERVAL_S = 30.0
MAX_CLIENT_FRAME_BYTES = 8192


class _CloseConnectionError(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


async def _authenticate(websocket: WebSocket, token: object) -> CurrentUser:
    if not isinstance(token, str) or not token:
        raise AuthenticationError("auth.token_missing", "Authentication is required.")
    async with session_scope(websocket.app) as session:
        return await resolve_access_token(session, token)


async def _audit_refusal(websocket: WebSocket, code: str, started: float) -> None:
    await write_audit_entry(
        websocket.app,
        AuditLog(
            user_id=None,
            ip_address=client_address(websocket.scope)[:45],
            method="WS",
            endpoint=websocket.url.path[:256],
            request_body_hash=None,
            response_status=401,
            duration_ms=int((time.perf_counter() - started) * 1000),
            timestamp_ms=int(time.time() * 1000),
            detail=None,
            outcome=f"refused: {code}",
        ),
    )


def _channels(value: object) -> set[Channel]:
    if not isinstance(value, list) or not value:
        raise _CloseConnectionError(
            CLOSE_BAD_REQUEST, "channels must be a non-empty list"
        )
    try:
        return {Channel(str(name)) for name in value}
    except ValueError as exc:
        raise _CloseConnectionError(CLOSE_BAD_REQUEST, "unknown channel") from exc


async def _receive_json(websocket: WebSocket) -> dict[str, Any]:
    text = await websocket.receive_text()
    if len(text) > MAX_CLIENT_FRAME_BYTES:
        raise _CloseConnectionError(CLOSE_BAD_REQUEST, "frame too large")
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CloseConnectionError(
            CLOSE_BAD_REQUEST, "frames must be JSON objects"
        ) from exc
    if not isinstance(message, dict):
        raise _CloseConnectionError(CLOSE_BAD_REQUEST, "frames must be JSON objects")
    return message


class _Connection:
    def __init__(
        self,
        websocket: WebSocket,
        hub: RealtimeHub,
        user: CurrentUser,
        token: str,
        subscriber: Subscriber,
    ) -> None:
        self.websocket = websocket
        self.hub = hub
        self.user = user
        self.token = token
        self.subscriber = subscriber

    async def read(self) -> None:
        while True:
            message = await _receive_json(self.websocket)
            kind = message.get("type")
            if kind == "subscribe":
                rate = message.get("max_rate_hz")
                if isinstance(rate, int | float) and rate > 0:
                    self.subscriber.max_rate_hz = min(float(rate), self.hub.max_rate_hz)
                self.hub.subscribe(self.subscriber, _channels(message.get("channels")))
                self._reply({"type": "subscribed", "channels": self._subscribed()})
            elif kind == "unsubscribe":
                self.hub.unsubscribe(
                    self.subscriber, _channels(message.get("channels"))
                )
                self._reply({"type": "subscribed", "channels": self._subscribed()})
            elif kind == "auth":
                await self._renew(message.get("access_token"))
            elif kind == "ping":
                self._reply({"type": "pong", "ts_ms": int(time.time() * 1000)})
            else:
                self._reply(
                    {
                        "type": "error",
                        "code": "ws.unknown_message",
                        "detail": "Expected subscribe, unsubscribe, auth or ping.",
                    }
                )

    async def write(self) -> None:
        while True:
            frame = await self.subscriber.queue.get()
            await self.websocket.send_text(frame)
            if self.subscriber.overflowed and self.subscriber.queue.empty():
                raise _CloseConnectionError(
                    CLOSE_TRY_AGAIN_LATER, "client too slow; reload state"
                )

    async def guard(self) -> None:
        """Close on token expiry, a closed session or a blocked account."""
        while True:
            now_ms = int(time.time() * 1000)
            wait_s = min(
                max((self.user.token_expires_at_ms - now_ms) / 1000.0, 0.0),
                REVALIDATE_INTERVAL_S,
            )
            await asyncio.sleep(wait_s)
            if self.subscriber.overflowed:
                raise _CloseConnectionError(
                    CLOSE_TRY_AGAIN_LATER, "client too slow; reload state"
                )
            if int(time.time() * 1000) >= self.user.token_expires_at_ms:
                raise _CloseConnectionError(CLOSE_UNAUTHORIZED, "token expired")
            await self._revalidate()

    async def _revalidate(self) -> None:
        try:
            self.user = await _authenticate(self.websocket, self.token)
        except AuthenticationError as exc:
            raise _CloseConnectionError(CLOSE_UNAUTHORIZED, exc.code) from exc

    async def _renew(self, token: object) -> None:
        try:
            renewed = await _authenticate(self.websocket, token)
        except AuthenticationError as exc:
            raise _CloseConnectionError(CLOSE_UNAUTHORIZED, exc.code) from exc
        if renewed.id != self.user.id:
            raise _CloseConnectionError(
                CLOSE_UNAUTHORIZED, "token belongs to another user"
            )
        self.user = renewed
        self.token = str(token)
        self._reply(
            {"type": "renewed", "token_expires_at_ms": renewed.token_expires_at_ms}
        )

    def _subscribed(self) -> list[str]:
        return sorted(channel.value for channel in self.subscriber.channels)

    def _reply(self, message: dict[str, Any]) -> None:
        self.subscriber.send_control(message)


@router.websocket("/ws")
async def realtime(websocket: WebSocket) -> None:
    hub: RealtimeHub | None = getattr(websocket.app.state, "realtime_hub", None)
    await websocket.accept()
    if hub is None:
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason="realtime unavailable")
        return

    started = time.perf_counter()
    try:
        first = await asyncio.wait_for(
            _receive_json(websocket), timeout=settings.ws_auth_timeout_s
        )
        if first.get("type") != "auth":
            raise AuthenticationError("auth.token_missing", "Authenticate first.")
        token = first.get("access_token")
        user = await _authenticate(websocket, token)
    except WebSocketDisconnect:
        return
    except (TimeoutError, AuthenticationError, _CloseConnectionError) as exc:
        if isinstance(exc, AuthenticationError):
            code = exc.code
        elif isinstance(exc, _CloseConnectionError):
            code = "ws.bad_request"
        else:
            code = "ws.auth_timeout"
        await _audit_refusal(websocket, code, started)
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=CLOSE_UNAUTHORIZED, reason=code)
        return

    subscriber = hub.register()
    connection = _Connection(websocket, hub, user, str(token), subscriber)
    subscriber.send_control(
        {
            "type": "welcome",
            "user": user.username,
            "role": user.role,
            "channels": [channel.value for channel in Channel],
            "max_rate_hz": subscriber.max_rate_hz,
            "token_expires_at_ms": user.token_expires_at_ms,
        }
    )
    actor = AuditActor(user.id, user.username, user.role)
    tasks = [
        asyncio.create_task(connection.read(), name=f"ws-read-{actor.username}"),
        asyncio.create_task(connection.write(), name=f"ws-write-{actor.username}"),
        asyncio.create_task(connection.guard(), name=f"ws-guard-{actor.username}"),
    ]
    close_code, close_reason = 1000, ""
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            error = task.exception()
            if isinstance(error, _CloseConnectionError):
                close_code, close_reason = error.code, error.reason
            elif error is not None and not isinstance(error, WebSocketDisconnect):
                logger.error(
                    "WebSocket session of %s failed", actor.username, exc_info=error
                )
                close_code, close_reason = 1011, "internal error"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        hub.unregister(subscriber)
        if (
            websocket.application_state is WebSocketState.CONNECTED
            and websocket.client_state is WebSocketState.CONNECTED
        ):
            with suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_text(
                    encode(
                        {"type": "closing", "code": close_code, "reason": close_reason}
                    )
                )
                await websocket.close(code=close_code, reason=close_reason)
