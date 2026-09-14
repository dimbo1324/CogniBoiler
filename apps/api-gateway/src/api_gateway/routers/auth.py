"""
Authentication endpoints backed by PostgreSQL.

POST /auth/login   — exchange username/password for JWT token pair
POST /auth/refresh — exchange a valid refresh token for a new access token
POST /auth/logout  — invalidate a refresh token in the DB blacklist
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from api_gateway.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
)
from api_gateway.auth.password import hash_password, verify_password
from api_gateway.dependencies import DbSession
from api_gateway.models.user import Role, TokenBlacklist, User, UserRole
from api_gateway.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_DUMMY_HASH = hash_password("cogniboiler-dummy-password")


async def _lookup_user_record(
    db: DbSession,
    username: str,
) -> tuple[User, str] | None:
    """Fetch an active user together with their primary RBAC role."""
    row = await db.execute(
        select(User, Role.name)
        .join(UserRole, UserRole.user_id == User.id)
        .join(Role, Role.id == UserRole.role_id)
        .where(User.username == username, User.is_active.is_(True))
        .limit(1)
    )
    result = row.first()
    if result is None:
        return None
    user, role_name = result
    return user, str(role_name)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    """Authenticate a user against PostgreSQL and return a JWT token pair."""
    user_record = await _lookup_user_record(db, body.username)
    stored_hash = user_record[0].hashed_password if user_record else _DUMMY_HASH
    password_ok = verify_password(body.password, stored_hash)

    if user_record is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user, role_name = user_record
    return TokenResponse(
        access_token=create_access_token(user.id, role_name),
        refresh_token=create_refresh_token(user.id, role_name),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbSession) -> TokenResponse:
    """Issue a new access token from a valid refresh token."""
    try:
        payload = decode_refresh_token(body.refresh_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    token_jti = str(payload.get("jti", ""))
    blacklisted = await db.scalar(
        select(TokenBlacklist).where(TokenBlacklist.token_jti == token_jti)
    )
    if blacklisted is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(str(payload["sub"]))
    role = str(payload["role"])
    return TokenResponse(
        access_token=create_access_token(user_id, role),
        refresh_token=create_refresh_token(user_id, role),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(body: LogoutRequest, db: DbSession) -> MessageResponse:
    """Blacklist a refresh token in PostgreSQL so it cannot be reused."""
    try:
        payload = decode_refresh_token(body.refresh_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    token_jti = str(payload.get("jti", ""))
    existing = await db.scalar(
        select(TokenBlacklist).where(TokenBlacklist.token_jti == token_jti)
    )
    if existing is None:
        db.add(
            TokenBlacklist(
                token_jti=token_jti,
                user_id=int(str(payload["sub"])),
                revoked_at_ms=_get_current_timestamp(),
                exp_ms=int(str(payload["exp"])) * 1000,
            )
        )
        await db.commit()

    return MessageResponse(message="Successfully logged out.")


def _get_current_timestamp() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time.time() * 1000)
