"""
The API gateway as the authority for OPC UA users and their writes.

An OPC UA user signs in with the same username and password as in the console. Every
method call is forwarded to the gateway's REST API with that user's access token, so the
gateway decides the role, audits the action under the user's name and reaches the PLC or
the alarm service exactly as the console does.

HTTP runs through urllib in a worker thread: the OPC UA server's event loop is never
blocked, and no HTTP library is added for a handful of calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "cogniboiler-opcua-server"
REQUEST_TIMEOUT_S = 10.0
REFRESH_MARGIN_MS = 30_000


@dataclass(frozen=True, slots=True)
class GatewayReply:
    status: int
    body: dict[str, Any]

    @property
    def detail(self) -> str:
        return str(self.body.get("detail") or self.body.get("reason") or "")


@dataclass(frozen=True, slots=True)
class GatewayTokens:
    access_token: str
    refresh_token: str
    access_expires_at_ms: int
    username: str
    role: str


class GatewayUnavailableError(ConnectionError):
    """The gateway did not answer."""


class GatewayClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None, token: str | None
    ) -> GatewayReply:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=data, method=method
        )
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", USER_AGENT)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return GatewayReply(response.status, _json(response.read()))
        except urllib.error.HTTPError as error:
            return GatewayReply(error.code, _json(error.read()))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GatewayUnavailableError(f"{method} {path}: {error}") from error

    async def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> GatewayReply:
        return await asyncio.to_thread(self._request, method, path, payload, token)

    async def login(self, username: str, password: str) -> GatewayTokens | None:
        reply = await self.request(
            "POST", "/auth/login", {"username": username, "password": password}
        )
        if reply.status != 200:
            logger.warning(
                "OPC UA sign-in of %r refused by the gateway: HTTP %d %s",
                username,
                reply.status,
                reply.body.get("code", ""),
            )
            return None
        return _tokens(reply.body)

    async def refresh(self, refresh_token: str) -> GatewayTokens | None:
        reply = await self.request(
            "POST", "/auth/refresh", {"refresh_token": refresh_token}
        )
        return _tokens(reply.body) if reply.status == 200 else None

    async def logout(self, refresh_token: str) -> None:
        await self.request("POST", "/auth/logout", {"refresh_token": refresh_token})


def _json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except UnicodeDecodeError, json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _tokens(body: dict[str, Any]) -> GatewayTokens | None:
    try:
        return GatewayTokens(
            access_token=str(body["access_token"]),
            refresh_token=str(body["refresh_token"]),
            access_expires_at_ms=int(body["access_expires_at_ms"]),
            username=str(body["username"]),
            role=str(body["role"]),
        )
    except KeyError, TypeError, ValueError:
        return None


class GatewaySession:
    """The gateway session of one OPC UA user: sign-in result, refreshed on demand."""

    def __init__(
        self, client: GatewayClient, login: asyncio.Task[GatewayTokens | None]
    ):
        self._client = client
        self._login = login
        self._tokens: GatewayTokens | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def tokens(self) -> GatewayTokens | None:
        """Valid tokens, refreshed when the access token is about to expire."""
        async with self._lock:
            if self._closed:
                return None
            if self._tokens is None:
                try:
                    self._tokens = await self._login
                except GatewayUnavailableError as exc:
                    logger.warning("OPC UA sign-in failed: %s", exc)
                    return None
                if self._tokens is None:
                    self._closed = True
                    return None
            now_ms = int(time.time() * 1000)
            if self._tokens.access_expires_at_ms - now_ms <= REFRESH_MARGIN_MS:
                self._tokens = await self._client.refresh(self._tokens.refresh_token)
                if self._tokens is None:
                    self._closed = True
            return self._tokens

    async def invalidate_access(self) -> None:
        """Force a refresh before the next call (the gateway rejected the token)."""
        async with self._lock:
            if self._tokens is not None:
                self._tokens = GatewayTokens(
                    access_token=self._tokens.access_token,
                    refresh_token=self._tokens.refresh_token,
                    access_expires_at_ms=0,
                    username=self._tokens.username,
                    role=self._tokens.role,
                )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if not self._login.done():
                self._login.cancel()
                return
            tokens = self._tokens
            if tokens is None and not self._login.cancelled():
                if self._login.exception() is None:
                    tokens = self._login.result()
            self._tokens = None
        if tokens is not None:
            try:
                await self._client.logout(tokens.refresh_token)
            except GatewayUnavailableError as exc:
                logger.info("Gateway sign-out skipped: %s", exc)
