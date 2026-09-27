"""
The audit trail.

What is recorded:
  - every mutating request (POST, PUT, PATCH, DELETE), sign-ins and sign-outs included;
  - every request refused with 401, 403 or 429, whatever its method;
  - reads of the audit log and of user accounts.
Telemetry reads are not recorded: the console polls them continuously and they change
nothing.

Each row holds who acted (id, name, and the role held at that moment), the client
address, the request, a SHA-256 digest of the body (never the body), the status, the
duration and the outcome a route reported — a PLC refusal is an HTTP 200 whose outcome
starts with "refused".

A body that carries a password (sign-in, password change, account creation, password
reset) leaves no digest at all: its username is known, so an unsalted digest of it is a
fast password verifier that anyone reading the log or a backup could crack offline.

The row is written after the response. If the write fails, the whole record goes to the
error log instead, so the trail survives in the service log.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from cogniboiler_runtime import now_ms
from fastapi import FastAPI, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api_gateway.dependencies import session_scope
from api_gateway.models.user import AuditLog
from api_gateway.observability import AUDIT_WRITE_FAILURES

logger = logging.getLogger(__name__)

_ACTOR_KEY = "audit_actor"
_OUTCOME_KEY = "audit_outcome"
_DETAIL_KEY = "audit_detail"

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
REFUSAL_STATUSES = frozenset({401, 403, 429})
AUDITED_READ_PREFIXES = ("/api/v1/audit", "/api/v1/users")
UNAUDITED_PATHS = frozenset({"/health", "/ready", "/docs", "/redoc", "/openapi.json"})
CREDENTIAL_PATHS = frozenset({"/auth/login", "/auth/password", "/api/v1/users"})
_CREDENTIAL_PATTERN = re.compile(r"/api/v1/users/[^/]+/password")


@dataclass(frozen=True, slots=True)
class AuditActor:
    user_id: int | None
    username: str | None
    role: str | None


def _state(scope: Scope) -> dict[str, Any]:
    state: dict[str, Any] = scope.setdefault("state", {})
    return state


def set_audit_actor(request: Request, actor: AuditActor) -> None:
    _state(request.scope)[_ACTOR_KEY] = actor


def set_audit_outcome(request: Request, outcome: str) -> None:
    _state(request.scope)[_OUTCOME_KEY] = outcome


def set_audit_detail(request: Request, detail: str) -> None:
    _state(request.scope)[_DETAIL_KEY] = detail


def command_outcome(accepted: bool, reason: str) -> str:
    return "accepted" if accepted else f"refused: {reason}"


def should_audit(method: str, path: str, status: int) -> bool:
    if path in UNAUDITED_PATHS:
        return False
    if method in MUTATING_METHODS or status in REFUSAL_STATUSES:
        return True
    return method == "GET" and path.startswith(AUDITED_READ_PREFIXES)


def carries_credentials(path: str) -> bool:
    """Whether a request body on this path can hold a password."""
    normalized = path.rstrip("/").lower()
    return (
        normalized in CREDENTIAL_PATHS
        or _CREDENTIAL_PATTERN.fullmatch(normalized) is not None
    )


def client_address(scope: Scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def _log_unstored(entry: AuditLog) -> None:
    AUDIT_WRITE_FAILURES.inc()
    logger.error(
        "Audit record NOT stored: %s %s status=%d user=%s(%s) role=%s ip=%s "
        "at_ms=%d outcome=%r detail=%r body_sha256=%s",
        entry.method,
        entry.endpoint,
        entry.response_status,
        entry.username,
        entry.user_id,
        entry.role,
        entry.ip_address,
        entry.timestamp_ms,
        entry.outcome,
        entry.detail,
        entry.request_body_hash,
        exc_info=True,
    )


async def write_audit_entry(app: FastAPI, entry: AuditLog) -> None:
    """Store one audit row; on failure log the whole record as an error."""
    try:
        async with session_scope(app) as session:
            session.add(entry)
            await session.commit()
    except asyncio.CancelledError:
        # Shutdown can cancel the write after the response went out: keep the record
        # in the log, then let the cancellation proceed.
        _log_unstored(entry)
        raise
    except Exception:
        _log_unstored(entry)


class AuditMiddleware:
    """ASGI middleware that records auditable HTTP requests once they are answered."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        received_at_ms = now_ms()
        started = time.perf_counter()
        digest = None if carries_credentials(str(scope["path"])) else hashlib.sha256()
        body_bytes = 0
        status = 500

        async def receive_and_hash() -> Message:
            nonlocal body_bytes
            message = await receive()
            if message["type"] == "http.request":
                chunk: bytes = message.get("body", b"")
                if chunk and digest is not None:
                    digest.update(chunk)
                    body_bytes += len(chunk)
            return message

        async def send_and_capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                elapsed = time.perf_counter() - started
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{elapsed:.4f}s".encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive_and_hash, send_and_capture)
        finally:
            method = str(scope["method"])
            path = str(scope["path"])
            if should_audit(method, path, status):
                state = _state(scope)
                actor: AuditActor = state.get(_ACTOR_KEY, AuditActor(None, None, None))
                query = scope.get("query_string", b"").decode("latin-1")
                detail = state.get(_DETAIL_KEY) or query or None
                await write_audit_entry(
                    scope["app"],
                    AuditLog(
                        user_id=actor.user_id,
                        username=actor.username,
                        role=actor.role,
                        ip_address=client_address(scope)[:45],
                        method=method,
                        endpoint=path[:256],
                        request_body_hash=(
                            digest.hexdigest() if digest and body_bytes else None
                        ),
                        response_status=status,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        timestamp_ms=received_at_ms,
                        detail=detail[:2000] if detail else None,
                        outcome=str(state.get(_OUTCOME_KEY, ""))[:500] or None,
                    ),
                )
