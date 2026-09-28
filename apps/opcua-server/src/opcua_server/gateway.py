"""
The API gateway as the authority for OPC UA users and their writes.

An OPC UA user signs in with the same username and password as in the console. Every
method call is forwarded to the gateway's REST API with that user's access token, so the
gateway decides the role, audits the action under the user's name and reaches the PLC or
the alarm service exactly as the console does.

HTTP runs through urllib in the client's own small thread pool: the OPC UA server's event
loop is never blocked, a flood of sign-ins cannot take the default executor from
everything else, and no HTTP library is added for a handful of calls. At most
MAX_CONCURRENT_LOGINS sign-ins run at once.

Requests made for a user carry the OPC UA peer address in X-Forwarded-For. The gateway
trusts that header from the Compose network, so its login throttle counts each OPC UA
client on its own and its audit rows name the client, not this server's container.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import http.client
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from typing import IO, Any, Protocol

from cogniboiler_observability import CORRELATION_HEADER, current_correlation_id
from cogniboiler_runtime import decode_json_object, now_ms

logger = logging.getLogger(__name__)

USER_AGENT = "cogniboiler-opcua-server"
REQUEST_TIMEOUT_S = 10.0
REFRESH_MARGIN_MS = 30_000
MAX_REPLY_BYTES = 1_048_576
ALLOWED_SCHEMES = frozenset({"http", "https"})
HTTP_WORKERS = 8
MAX_CONCURRENT_LOGINS = 4
FORWARDED_FOR_HEADER = "X-Forwarded-For"


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is an answer, not an instruction: it would carry the bearer token."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class GatewayClient:
    def __init__(self, base_url: str) -> None:
        if urllib.parse.urlsplit(base_url).scheme not in ALLOWED_SCHEMES:
            raise ValueError("the gateway URL must start with http:// or https://")
        self._base_url = base_url.rstrip("/")
        # No proxy from the environment and no redirects: the user's password and
        # access token go to the configured gateway and nowhere else.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        self._executor = ThreadPoolExecutor(
            max_workers=HTTP_WORKERS, thread_name_prefix="opcua-gateway"
        )
        self._logins = asyncio.Semaphore(MAX_CONCURRENT_LOGINS)

    def close(self) -> None:
        """Stop the worker threads; requests not yet started are dropped."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        token: str | None,
        client_address: str | None = None,
    ) -> GatewayReply:
        data = (
            None
            if payload is None
            else json.dumps(payload, allow_nan=False).encode("utf-8")
        )
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=data, method=method
        )
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", USER_AGENT)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        correlation_id = current_correlation_id()
        if correlation_id:
            request.add_header(CORRELATION_HEADER, correlation_id)
        if client_address:
            request.add_header(FORWARDED_FOR_HEADER, client_address)
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
                return GatewayReply(response.status, _body(response, method, path))
        except urllib.error.HTTPError as error:
            with error:
                return GatewayReply(error.code, _body(error, method, path))
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            # HTTPException: a reply cut short or a broken status line while the
            # gateway restarts; ValueError: a header value http.client refuses.
            raise GatewayUnavailableError(
                f"{method} {path}: {type(error).__name__}: {error}"
            ) from error

    async def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
        client_address: str | None = None,
    ) -> GatewayReply:
        call = functools.partial(
            self._request, method, path, payload, token, client_address
        )
        # run_in_executor, unlike to_thread, does not carry the context over: the
        # correlation id lives in a context variable.
        context = contextvars.copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, context.run, call
        )

    async def login(
        self, username: str, password: str, client_address: str | None = None
    ) -> GatewayTokens | None:
        async with self._logins:
            reply = await self.request(
                "POST",
                "/auth/login",
                {"username": username, "password": password},
                client_address=client_address,
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

    async def refresh(
        self, refresh_token: str, client_address: str | None = None
    ) -> GatewayTokens | None:
        reply = await self.request(
            "POST",
            "/auth/refresh",
            {"refresh_token": refresh_token},
            client_address=client_address,
        )
        return _tokens(reply.body) if reply.status == 200 else None

    async def logout(
        self, refresh_token: str, client_address: str | None = None
    ) -> None:
        await self.request(
            "POST",
            "/auth/logout",
            {"refresh_token": refresh_token},
            client_address=client_address,
        )


class _Reply(Protocol):
    headers: Message

    def read(self, amt: int, /) -> bytes: ...


def _body(response: _Reply, method: str, path: str) -> dict[str, Any]:
    raw = response.read(MAX_REPLY_BYTES + 1)
    if len(raw) > MAX_REPLY_BYTES:
        logger.warning(
            "Gateway reply to %s %s is larger than %d bytes; ignored",
            method,
            path,
            MAX_REPLY_BYTES,
        )
        return {}
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and len(raw) < int(declared):
        # read(n) returns what arrived before the connection closed, silently.
        raise http.client.IncompleteRead(raw, int(declared) - len(raw))
    return decode_json_object(raw, max_bytes=MAX_REPLY_BYTES) or {}


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
        self,
        client: GatewayClient,
        login: asyncio.Task[GatewayTokens | None],
        client_address: str | None = None,
    ):
        self._client = client
        self._login = login
        self._client_address = client_address
        self._tokens: GatewayTokens | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def client_address(self) -> str | None:
        """The OPC UA peer this session speaks for, as the gateway should record it."""
        return self._client_address

    async def tokens(self) -> GatewayTokens | None:
        """Valid tokens, refreshed when the access token is about to expire."""
        async with self._lock:
            if self._closed:
                return None
            if self._tokens is None:
                try:
                    self._tokens = await self._login
                except Exception as exc:
                    # The password went with the failed attempt: the session cannot
                    # sign in again, so it says so once and stays closed.
                    logger.warning(
                        "OPC UA sign-in failed: %s: %s", type(exc).__name__, exc
                    )
                    self._closed = True
                    return None
                if self._tokens is None:
                    self._closed = True
                    return None
            if self._tokens.access_expires_at_ms - now_ms() <= REFRESH_MARGIN_MS:
                self._tokens = await self._client.refresh(
                    self._tokens.refresh_token, client_address=self._client_address
                )
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
        """Sign out at the gateway, once: a session already closed has nothing to end."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tokens = self._tokens
            self._tokens = None
        if tokens is None:
            # A sign-in still under way is waited for, not cancelled: its request is
            # already in a worker thread, and the gateway would open a session that
            # nobody signs out.
            tokens = await self._login_result()
        if tokens is not None:
            try:
                await self._client.logout(
                    tokens.refresh_token, client_address=self._client_address
                )
            except GatewayUnavailableError as exc:
                logger.info("Gateway sign-out skipped: %s", exc)

    async def _login_result(self) -> GatewayTokens | None:
        if self._login.cancelled():
            return None
        try:
            return await self._login
        except Exception:
            return None
