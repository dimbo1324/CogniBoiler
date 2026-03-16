"""
Authentication endpoints.

POST /auth/login   — exchange username/password for JWT token pair
POST /auth/refresh — exchange a valid refresh token for a new access token
POST /auth/logout  — invalidate a refresh token (server-side blacklist)

No database is wired yet — user lookup uses an in-memory stub so the
routing, validation, and JWT logic can be tested without PostgreSQL.
The stub will be replaced with a real SQLAlchemy query in Phase 5.4.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from api_gateway.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
)
from api_gateway.auth.password import verify_password
from api_gateway.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# ─── In-memory user stub ─────────────────────────────────────────────────────
# Temporary fixture for Phase 5.1 — replaced by PostgreSQL in Phase 5.4.
# Passwords are stored as Argon2id hashes (generated with hash_password()).
_STUB_USERS: dict[str, dict[str, str]] = {
    "admin": {
        "id": "1",
        "role": "admin",
        "hashed_password": "$argon2id$v=19$m=65536,t=2,p=2$stub_admin",
    },
    "operator": {
        "id": "2",
        "role": "operator",
        "hashed_password": "$argon2id$v=19$m=65536,t=2,p=2$stub_operator",
    },
}

# ─── In-memory refresh token blacklist ───────────────────────────────────────
# In Phase 5.4 this moves to a PostgreSQL table with TTL cleanup.
_blacklisted_tokens: set[str] = set()


def _lookup_user(username: str) -> dict[str, str] | None:
    """Return the stub user record or None if not found."""
    return _STUB_USERS.get(username)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)  # type: ignore[misc]
async def login(body: LoginRequest) -> TokenResponse:
    """
    Authenticate a user and return a JWT token pair.

    Returns 401 for both unknown username and wrong password — the same
    error message prevents username enumeration attacks.
    """
    user = _lookup_user(body.username)

    # Constant-time path: always call verify_password even if user is None
    # to prevent timing-based username enumeration.
    stored_hash = user["hashed_password"] if user else "$argon2id$v=19$dummy"
    password_ok = verify_password(body.password, stored_hash)

    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(user["id"])
    role = user["role"]

    return TokenResponse(
        access_token=create_access_token(user_id, role),
        refresh_token=create_refresh_token(user_id, role),
    )


@router.post("/refresh", response_model=TokenResponse)  # type: ignore[misc]
async def refresh(body: RefreshRequest) -> TokenResponse:
    """
    Issue a new access token from a valid refresh token.

    Returns 401 if the token is expired, invalid, or blacklisted.
    """
    if body.refresh_token in _blacklisted_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_refresh_token(body.refresh_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = int(str(payload["sub"]))
    role = str(payload["role"])

    return TokenResponse(
        access_token=create_access_token(user_id, role),
        refresh_token=create_refresh_token(user_id, role),
    )


@router.post("/logout", response_model=MessageResponse)  # type: ignore[misc]
async def logout(body: LogoutRequest) -> MessageResponse:
    """
    Invalidate a refresh token by adding it to the blacklist.

    Always returns 200 even if the token was already blacklisted,
    to prevent information leakage about token state.
    """
    _blacklisted_tokens.add(body.refresh_token)
    return MessageResponse(message="Successfully logged out.")


def _get_current_timestamp() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time.time() * 1000)
