"""The API gateway client `smoke` and `demo` share: stdlib urllib, JSON in and out.

Both scripts talk to the gateway the way the console does and sign in with the demo
accounts from `.env`. One copy here keeps a change to timeouts, headers or error handling
from being made twice — the two copies had already drifted, and one of them threw away
the reason of every refusal.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .envfile import EnvFileError, parse, values


class GatewayUnreachableError(RuntimeError):
    """The gateway did not answer at all."""


def dig(body: Any, *keys: str) -> Any:
    """Follow ``keys`` through nested JSON objects; ``None`` as soon as one is missing."""
    for key in keys:
        if not isinstance(body, dict):
            return None
        body = body.get(key)
    return body


@dataclass(frozen=True)
class Reply:
    status: int
    body: Any

    @property
    def accepted(self) -> bool:
        return self.status == 200 and bool(dig(self.body, "accepted"))

    @property
    def refusal(self) -> str:
        reason = dig(self.body, "reason") or dig(self.body, "detail")
        return f"HTTP {self.status}" + (f": {reason}" if reason else "")


def _decoded(raw: bytes) -> Any:
    return json.loads(raw) if raw else None


def call(
    base_url: str,
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> Reply:
    """One request. An HTTP error is a ``Reply`` that keeps its body (the problem
    details say why); no answer at all raises ``GatewayUnreachableError``."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base_url}{path}", data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Reply(response.status, _decoded(response.read()))
    except urllib.error.HTTPError as error:
        try:
            return Reply(error.code, _decoded(error.read()))
        except ValueError:
            return Reply(error.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise GatewayUnreachableError(f"{method} {path}: {error}") from error


def sign_in(base_url: str, username: str, password: str) -> tuple[str | None, Reply]:
    """The access token, or ``None`` with the reply that refused it."""
    reply = call(
        base_url,
        "POST",
        "/auth/login",
        payload={"username": username, "password": password},
    )
    token = dig(reply.body, "access_token")
    return (token if isinstance(token, str) else None), reply


def load_env(root: Path, env_file: str) -> dict[str, str]:
    """The values of the ``.env`` the stack was started with.

    Every problem is one ``EnvFileError`` whose message starts with the file name.
    """
    path = root / env_file
    if not path.is_file():
        raise EnvFileError(
            f"{env_file} is missing — the stack cannot have been started"
        )
    try:
        return values(parse(path.read_text(encoding="utf-8")))
    except EnvFileError as error:
        raise EnvFileError(f"{env_file}: {error}") from error
