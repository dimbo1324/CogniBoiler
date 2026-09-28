"""A stand-in API gateway over real HTTP, and a recorder of address-space writes."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PASSWORD = "operator-password-1"


@dataclass
class Call:
    method: str
    path: str
    body: dict[str, Any]
    authorization: str
    correlation_id: str
    forwarded_for: str = ""


@dataclass
class GatewayScript:
    """What the stand-in gateway answers; tests change it between calls."""

    calls: list[Call] = field(default_factory=list)
    access_ttl_ms: int = 900_000
    command_status: int = 200
    command_body: dict[str, Any] = field(
        default_factory=lambda: {"accepted": True, "reason": ""}
    )
    reject_access_once: bool = False
    refresh_works: bool = True
    raw_command_body: bytes | None = None
    redirect_commands_to: str | None = None
    truncate_command_body: bool = False
    issued: int = 0

    def tokens(self, username: str) -> dict[str, Any]:
        self.issued += 1
        return {
            "access_token": f"access-{self.issued}",
            "refresh_token": f"refresh-{self.issued}",
            "access_expires_at_ms": int(time.time() * 1000) + self.access_ttl_ms,
            "username": username,
            "role": "operator",
        }


def _handler(script: GatewayScript) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return None

        def _reply(self, status: int, body: dict[str, Any] | bytes) -> None:
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _record(self, method: str, body: dict[str, Any]) -> None:
            script.calls.append(
                Call(
                    method,
                    self.path,
                    body,
                    self.headers.get("Authorization", ""),
                    self.headers.get("X-Correlation-ID", ""),
                    self.headers.get("X-Forwarded-For", ""),
                )
            )

        def do_GET(self) -> None:  # noqa: N802
            self._record("GET", {})
            self._reply(200, {"accepted": True, "reason": "redirected"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw else {}
            self._record("POST", body)
            if self.path == "/auth/login":
                if body.get("password") == PASSWORD:
                    self._reply(200, script.tokens(str(body["username"])))
                else:
                    self._reply(401, {"code": "auth.invalid_credentials"})
            elif self.path == "/auth/refresh":
                if script.refresh_works:
                    self._reply(200, script.tokens("operator1"))
                else:
                    self._reply(401, {"code": "auth.refresh_invalid"})
            elif self.path == "/auth/logout":
                self._reply(200, {"message": "Signed out."})
            elif script.reject_access_once:
                script.reject_access_once = False
                self._reply(401, {"code": "auth.token_expired"})
            elif script.redirect_commands_to is not None:
                self.send_response(303)
                self.send_header("Location", script.redirect_commands_to)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif script.truncate_command_body:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.write(b'{"accepted": tr')
            elif script.raw_command_body is not None:
                self._reply(script.command_status, script.raw_command_body)
            else:
                self._reply(script.command_status, script.command_body)

    return Handler


@contextmanager
def gateway_server() -> Iterator[tuple[str, GatewayScript]]:
    script = GatewayScript()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(script))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", script
    finally:
        server.shutdown()
        server.server_close()


class RecordingOPC:
    """The update side of CogniBoilerOPCServer, recording every write."""

    def __init__(self) -> None:
        self.updates: list[tuple[int, Any, int]] = []
        self.stale: list[int] = []
        self.missing: set[int] = set()

    async def update_variable(
        self,
        node_id: int,
        value: Any,
        *,
        quality: int = 0,
        source_timestamp_ms: int | None = None,
    ) -> None:
        if node_id in self.missing:
            raise KeyError(node_id)
        self.updates.append((node_id, value, quality))

    async def mark_stale(self, node_ids: Iterable[int]) -> None:
        self.stale.extend(node_ids)

    def latest(self, node_id: int) -> Any:
        return next(value for nid, value, _ in reversed(self.updates) if nid == node_id)
