"""
Response headers the gateway sets itself rather than leaving to nginx.

A host-run gateway behind the Vite proxy has no nginx in front of it, and token responses
must not be cached anywhere (RFC 6749 §5.1). So:
  - every response under /auth, /api/v1/users and /api/v1/audit — tokens, accounts and
    the audit log — carries Cache-Control: no-store and Pragma: no-cache;
  - every response carries X-Content-Type-Options: nosniff.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

NO_STORE_PREFIXES = ("/auth", "/api/v1/users", "/api/v1/audit")

_NOSNIFF = (b"x-content-type-options", b"nosniff")
_NO_STORE = ((b"cache-control", b"no-store"), (b"pragma", b"no-cache"))


def _no_store(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in NO_STORE_PREFIXES
    )


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = [_NOSNIFF]
        if _no_store(str(scope["path"])):
            extra.extend(_NO_STORE)
        replaced = {name for name, _ in extra}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                kept = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if bytes(name).lower() not in replaced
                ]
                message["headers"] = kept + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)
