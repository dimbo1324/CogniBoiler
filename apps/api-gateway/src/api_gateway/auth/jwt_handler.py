"""
JWT token creation and verification using RS256 (RSA-2048).

Why RS256 over HS256:
  - HS256 uses a single shared secret for both signing and verification.
    Every service that needs to verify tokens must know the secret.
  - RS256 uses an asymmetric key pair: the private key signs tokens
    (only the gateway holds it), the public key verifies them
    (can be shared freely with any downstream service).

Token types:
  - access  : short-lived (15 min), used in Authorization header
  - refresh : 7 days from sign-in, used only at POST /auth/refresh, rotated on use

Payload structure:
    {
        "sub":  "42",           # user ID as string
        "role": "operator",     # role at issue time; the database stays authoritative
        "type": "access",       # "access" | "refresh"
        "jti":  "<uuid>",       # token id; refresh tokens are stored by it
        "sid":  "<uuid>",       # session: the refresh-token family of one sign-in
        "iat":  1710000000,
        "exp":  1710000900,
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

import jwt
from cogniboiler_runtime import MILLISECONDS_PER_DAY

from api_gateway.config import settings

# ─── Token payload type alias ────────────────────────────────────────────────

TokenData = dict[str, str | int]


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """A signed token together with the claims the server stores or reports."""

    token: str
    jti: str
    session_id: str
    issued_at_ms: int
    expires_at_ms: int


# ─── Key helpers ─────────────────────────────────────────────────────────────


def _private_key() -> str:
    """
    Return the RSA private key PEM string from settings.

    Raises RuntimeError if the key is not configured.
    """
    if not settings.jwt_private_key:
        raise RuntimeError(
            "jwt_private_key is not set. Run "
            "`python dev_tools_scripts_runner.py dev-secrets` and set "
            "JWT_PRIVATE_KEY in your .env file."
        )
    return settings.jwt_private_key


def _public_key() -> str:
    """
    Return the RSA public key PEM string from settings.

    Raises RuntimeError if the key is not configured.
    """
    if not settings.jwt_public_key:
        raise RuntimeError(
            "jwt_public_key is not set. Set JWT_PUBLIC_KEY in your .env file."
        )
    return settings.jwt_public_key


# ─── Token creation ───────────────────────────────────────────────────────────


def _issue(
    *,
    user_id: int,
    role: str,
    token_type: str,
    session_id: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> IssuedToken:
    jti = str(uuid4())
    payload: TokenData = {
        "sub": str(user_id),
        "role": role,
        "type": token_type,
        "jti": jti,
        "sid": session_id,
        "iat": issued_at_ms // 1000,
        "exp": expires_at_ms // 1000,
    }
    token = jwt.encode(payload, _private_key(), algorithm=settings.jwt_algorithm)
    return IssuedToken(
        token=token,
        jti=jti,
        session_id=session_id,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
    )


def issue_access_token(
    user_id: int,
    role: str,
    session_id: str,
    *,
    not_after_ms: int | None = None,
) -> IssuedToken:
    """
    Sign an access token for a session.

    It expires after jwt_access_token_expire_minutes, but never after not_after_ms —
    the session's own expiry — so no access token outlives its sign-in.
    """
    now_ms = int(time.time() * 1000)
    expires_at_ms = now_ms + settings.jwt_access_token_expire_minutes * 60_000
    if not_after_ms is not None:
        expires_at_ms = min(expires_at_ms, not_after_ms)
    return _issue(
        user_id=user_id,
        role=role,
        token_type="access",
        session_id=session_id,
        issued_at_ms=now_ms,
        expires_at_ms=expires_at_ms,
    )


def issue_refresh_token(
    user_id: int,
    role: str,
    session_id: str,
    *,
    expires_at_ms: int | None = None,
) -> IssuedToken:
    """
    Sign a refresh token for a session.

    A new session expires jwt_refresh_token_expire_days after sign-in; a rotated
    token passes the session's expiry in, so rotation never extends a session.
    """
    now_ms = int(time.time() * 1000)
    if expires_at_ms is None:
        expires_at_ms = (
            now_ms + settings.jwt_refresh_token_expire_days * MILLISECONDS_PER_DAY
        )
    return _issue(
        user_id=user_id,
        role=role,
        token_type="refresh",
        session_id=session_id,
        issued_at_ms=now_ms,
        expires_at_ms=expires_at_ms,
    )


def create_access_token(user_id: int, role: str, session_id: str = "") -> str:
    """Signed access token string; the gateway accepts it only for a stored session."""
    return issue_access_token(user_id, role, session_id or str(uuid4())).token


def create_refresh_token(user_id: int, role: str, session_id: str = "") -> str:
    """Signed refresh token string; the gateway exchanges it only if it stored it."""
    return issue_refresh_token(user_id, role, session_id or str(uuid4())).token


# ─── Token verification ───────────────────────────────────────────────────────


def decode_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Performs full RS256 signature verification and expiry check.

    Raises:
        jwt.ExpiredSignatureError: Token has expired.
        jwt.InvalidTokenError:     Signature invalid or malformed token.
    """
    return jwt.decode(
        token,
        _public_key(),
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "iat", "sub", "jti", "type"]},
    )


def decode_access_token(token: str) -> TokenData:
    """
    Verify and decode an access token.

    Same as decode_token() but additionally checks that the token type
    is "access". Rejects refresh tokens used in place of access tokens.

    Raises:
        jwt.InvalidTokenError: Wrong token type or verification failure.
    """
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Expected access token, got refresh token.")
    return payload


def decode_refresh_token(token: str) -> TokenData:
    """
    Verify and decode a refresh token.

    Same as decode_token() but additionally checks that the token type
    is "refresh". Rejects access tokens used at the refresh endpoint.

    Raises:
        jwt.InvalidTokenError: Wrong token type or verification failure.
    """
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise jwt.InvalidTokenError("Expected refresh token, got access token.")
    return payload
