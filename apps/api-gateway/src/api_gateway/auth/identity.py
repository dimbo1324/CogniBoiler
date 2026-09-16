"""
Who is calling: roles, accounts and the resolution of an access token to a user.

An access token proves only that the gateway issued it. Every request also checks the
database: the account must still be active, the token's session must still be open, and
the role is read from the account — so blocking a user, closing a session or changing a
role takes effect on the next request, not when the token expires.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import jwt
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.auth.jwt_handler import decode_access_token
from api_gateway.models.user import RefreshToken, Role, User, UserRole

RoleName = Literal["viewer", "operator", "engineer", "admin"]

ROLE_HIERARCHY: tuple[RoleName, ...] = ("viewer", "operator", "engineer", "admin")


def role_level(role: str) -> int:
    """Privilege level: 0 (viewer) … 3 (admin); -1 for an unknown or missing role."""
    try:
        return ROLE_HIERARCHY.index(cast(RoleName, role))
    except ValueError:
        return -1


def effective_role(names: Iterable[str | None]) -> str:
    """The most privileged known role of an account, or "" when it has none."""
    known = [name for name in names if name is not None and role_level(name) >= 0]
    return max(known, key=role_level, default="")


@dataclass(frozen=True, slots=True)
class Account:
    """An account with its effective role, as stored."""

    id: int
    username: str
    hashed_password: str
    is_active: bool
    role: str
    created_at_ms: int
    last_login_at_ms: int | None

    @property
    def can_sign_in(self) -> bool:
        return self.is_active and role_level(self.role) >= 0


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """The authenticated caller of one request."""

    id: int
    username: str
    role: str
    session_id: str
    token_expires_at_ms: int

    def at_least(self, role: str) -> bool:
        return role_level(self.role) >= role_level(role) >= 0


class AuthenticationError(Exception):
    """An access token that does not identify an active user in an open session."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


async def load_account(
    db: AsyncSession, *, user_id: int | None = None, username: str | None = None
) -> Account | None:
    """One account with its effective role, by id or by name."""
    statement = (
        select(User, Role.name)
        .outerjoin(UserRole, UserRole.user_id == User.id)
        .outerjoin(Role, Role.id == UserRole.role_id)
    )
    if user_id is not None:
        statement = statement.where(User.id == user_id)
    elif username is not None:
        statement = statement.where(User.username == username)
    else:
        raise ValueError("load_account needs a user_id or a username")
    rows = (await db.execute(statement)).all()
    if not rows:
        return None
    user: User = rows[0][0]
    return Account(
        id=user.id,
        username=user.username,
        hashed_password=user.hashed_password,
        is_active=user.is_active,
        role=effective_role(row[1] for row in rows),
        created_at_ms=user.created_at_ms,
        last_login_at_ms=user.last_login_at_ms,
    )


async def resolve_access_token(db: AsyncSession, token: str) -> CurrentUser:
    """
    The user behind an access token.

    Raises:
        AuthenticationError: expired or invalid token, closed or unknown session,
            blocked account, or an account without a role.
    """
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError(
            "auth.token_expired", "The access token has expired."
        ) from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError(
            "auth.token_invalid", "The access token is invalid."
        ) from exc

    session_id = str(payload.get("sid", ""))
    try:
        user_id = int(str(payload["sub"]))
    except (KeyError, ValueError) as exc:
        raise AuthenticationError(
            "auth.token_invalid", "The access token is invalid."
        ) from exc
    if not session_id:
        raise AuthenticationError("auth.token_invalid", "The access token is invalid.")

    session_known = exists().where(
        RefreshToken.family_id == session_id, RefreshToken.user_id == user_id
    )
    session_closed = exists().where(
        RefreshToken.family_id == session_id,
        RefreshToken.revoked_at_ms.is_not(None),
    )
    rows = (
        await db.execute(
            select(
                User.username,
                User.is_active,
                Role.name,
                session_known.label("known"),
                session_closed.label("closed"),
            )
            .outerjoin(UserRole, UserRole.user_id == User.id)
            .outerjoin(Role, Role.id == UserRole.role_id)
            .where(User.id == user_id)
        )
    ).all()
    if not rows:
        raise AuthenticationError("auth.session_invalid", "The session is not valid.")
    username, is_active, _, known, closed = rows[0]
    role = effective_role(row[2] for row in rows)
    if not known or closed or not is_active or role_level(role) < 0:
        raise AuthenticationError("auth.session_invalid", "The session is not valid.")
    return CurrentUser(
        id=user_id,
        username=str(username),
        role=role,
        session_id=session_id,
        token_expires_at_ms=int(str(payload["exp"])) * 1000,
    )
