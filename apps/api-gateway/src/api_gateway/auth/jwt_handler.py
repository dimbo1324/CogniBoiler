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
  - refresh : long-lived (7 days), used only at POST /auth/refresh

Payload structure:
    {
        "sub":  "42",           # user ID as string
        "role": "operator",     # RBAC role
        "type": "access",       # "access" | "refresh"
        "iat":  1710000000,     # issued-at  (added by PyJWT)
        "exp":  1710000900,     # expiry     (added by PyJWT)
    }
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from api_gateway.config import settings

# ─── Token payload type alias ────────────────────────────────────────────────

TokenData = dict[str, str | int]


# ─── Key helpers ─────────────────────────────────────────────────────────────


def _private_key() -> str:
    """
    Return the RSA private key PEM string from settings.

    Raises RuntimeError if the key is not configured.
    """
    if not settings.jwt_private_key:
        raise RuntimeError(
            "jwt_private_key is not set. "
            "Generate a key pair with scripts/gen_keys.py and set "
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
            "jwt_public_key is not set. " "Set JWT_PUBLIC_KEY in your .env file."
        )
    return settings.jwt_public_key


# ─── Token creation ───────────────────────────────────────────────────────────


def create_access_token(user_id: int, role: str) -> str:
    """
    Create a short-lived RS256 access token.

    The token expires in jwt_access_token_expire_minutes (default: 15 min).
    It is intended to be sent in the Authorization: Bearer <token> header.

    Args:
        user_id: Database primary key of the authenticated user.
        role:    RBAC role string (e.g. "operator", "admin").

    Returns:
        Signed JWT string.
    """
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload: TokenData = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
    }

    return jwt.encode(  # type: ignore[no-any-return]
        payload, _private_key(), algorithm=settings.jwt_algorithm
    )


def create_refresh_token(user_id: int, role: str) -> str:
    """
    Create a long-lived RS256 refresh token.

    The token expires in jwt_refresh_token_expire_days (default: 7 days).
    It must be stored server-side (PostgreSQL) and invalidated after use
    (rotation) or on logout (blacklist).

    Args:
        user_id: Database primary key of the authenticated user.
        role:    RBAC role string.

    Returns:
        Signed JWT string.
    """
    now = datetime.now(UTC)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload: TokenData = {
        "sub": str(user_id),
        "role": role,
        "type": "refresh",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
    }

    return jwt.encode(  # type: ignore[no-any-return]
        payload, _private_key(), algorithm=settings.jwt_algorithm
    )


# ─── Token verification ───────────────────────────────────────────────────────


def decode_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Performs full RS256 signature verification and expiry check.

    Args:
        token: Raw JWT string from the Authorization header.

    Returns:
        Decoded payload dict with keys: sub, role, type, exp, iat.

    Raises:
        jwt.ExpiredSignatureError: Token has expired.
        jwt.InvalidTokenError:     Signature invalid or malformed token.
    """
    return jwt.decode(  # type: ignore[no-any-return]
        token,
        _public_key(),
        algorithms=[settings.jwt_algorithm],
    )


def decode_access_token(token: str) -> TokenData:
    """
    Verify and decode an access token.

    Same as decode_token() but additionally checks that the token type
    is "access". Rejects refresh tokens used in place of access tokens.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

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

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        jwt.InvalidTokenError: Wrong token type or verification failure.
    """
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise jwt.InvalidTokenError("Expected refresh token, got access token.")
    return payload
