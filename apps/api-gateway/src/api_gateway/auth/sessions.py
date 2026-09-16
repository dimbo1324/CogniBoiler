"""
Sign-in sessions: refresh-token families with rotation and revocation.

- A sign-in opens a family and stores its first refresh token.
- A refresh exchanges the presented token for a successor in the same family. The
  successor keeps the family's absolute expiry, so using a session never extends it.
- A token presented again after it was exchanged means it leaked or was replayed: the
  whole family is revoked. Within a short grace period (two tabs refreshing at once) the
  late request is only refused, and the session survives.
- Sign-out, a password change, a role change or blocking the account revoke families;
  access tokens name their family, so they stop working on the next request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from uuid import uuid4

import jwt
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.auth.identity import Account, load_account
from api_gateway.auth.jwt_handler import (
    IssuedToken,
    decode_refresh_token,
    issue_access_token,
    issue_refresh_token,
)
from api_gateway.config import settings
from api_gateway.models.user import RefreshToken, User

logger = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class SessionTokens:
    access: IssuedToken
    refresh: IssuedToken
    account: Account


class RefreshRejectedError(Exception):
    """A refresh token that cannot be exchanged."""

    def __init__(self, code: str, detail: str, *, session_revoked: bool) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.session_revoked = session_revoked


@dataclass(frozen=True, slots=True)
class ClientInfo:
    ip: str
    user_agent: str

    @property
    def stored_user_agent(self) -> str | None:
        return self.user_agent[:256] or None


async def open_session(
    db: AsyncSession, account: Account, client: ClientInfo
) -> SessionTokens:
    """Start a session for a verified account and record the sign-in. Commits."""
    session_id = str(uuid4())
    refresh = issue_refresh_token(account.id, account.role, session_id)
    access = issue_access_token(
        account.id, account.role, session_id, not_after_ms=refresh.expires_at_ms
    )
    db.add(
        RefreshToken(
            jti=refresh.jti,
            family_id=session_id,
            user_id=account.id,
            issued_at_ms=refresh.issued_at_ms,
            expires_at_ms=refresh.expires_at_ms,
            client_ip=client.ip,
            user_agent=client.stored_user_agent,
        )
    )
    await db.execute(
        update(User)
        .where(User.id == account.id)
        .values(last_login_at_ms=refresh.issued_at_ms)
    )
    await db.commit()
    return SessionTokens(access=access, refresh=refresh, account=account)


async def rotate_session(
    db: AsyncSession, token: str, client: ClientInfo
) -> SessionTokens:
    """
    Exchange a refresh token for a new token pair. Commits.

    Raises:
        RefreshRejectedError: invalid, expired, unknown, revoked or reused token, or an
            account that can no longer sign in.
    """
    invalid = RefreshRejectedError(
        "auth.refresh_invalid",
        "The refresh token is invalid or expired.",
        session_revoked=False,
    )
    try:
        payload = decode_refresh_token(token)
    except jwt.PyJWTError as exc:
        raise invalid from exc

    record = await db.scalar(
        select(RefreshToken)
        .where(RefreshToken.jti == str(payload.get("jti", "")))
        .with_for_update()
    )
    if record is None or str(record.user_id) != str(payload.get("sub")):
        raise invalid
    current_ms = now_ms()
    if record.revoked_at_ms is not None or record.expires_at_ms <= current_ms:
        raise invalid

    if record.used_at_ms is not None:
        if current_ms - record.used_at_ms <= settings.refresh_reuse_grace_s * 1000:
            raise RefreshRejectedError(
                "auth.refresh_superseded",
                "The refresh token was already exchanged; use its successor.",
                session_revoked=False,
            )
        revoked = await revoke_family(db, record.family_id, "reuse")
        await db.commit()
        logger.warning(
            "Refresh token reused: session %s of user %d revoked (%d tokens), "
            "client %s",
            record.family_id,
            record.user_id,
            revoked,
            client.ip,
        )
        raise RefreshRejectedError(
            "auth.refresh_reused",
            "The refresh token was already used; the session has been closed.",
            session_revoked=True,
        )

    account = await load_account(db, user_id=record.user_id)
    if account is None or not account.can_sign_in:
        await revoke_family(db, record.family_id, "blocked")
        await db.commit()
        raise invalid

    refresh = issue_refresh_token(
        account.id,
        account.role,
        record.family_id,
        expires_at_ms=record.expires_at_ms,
    )
    access = issue_access_token(
        account.id, account.role, record.family_id, not_after_ms=record.expires_at_ms
    )
    record.used_at_ms = current_ms
    record.replaced_by_jti = refresh.jti
    db.add(
        RefreshToken(
            jti=refresh.jti,
            family_id=record.family_id,
            user_id=account.id,
            issued_at_ms=refresh.issued_at_ms,
            expires_at_ms=refresh.expires_at_ms,
            client_ip=client.ip,
            user_agent=client.stored_user_agent,
        )
    )
    await db.commit()
    return SessionTokens(access=access, refresh=refresh, account=account)


async def revoke_family(db: AsyncSession, session_id: str, reason: str) -> int:
    """Close one session. Does not commit; returns the number of tokens revoked."""
    result = await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.family_id == session_id,
            RefreshToken.revoked_at_ms.is_(None),
        )
        .values(revoked_at_ms=now_ms(), revoked_reason=reason)
    )
    return int(getattr(result, "rowcount", 0) or 0)


async def revoke_user_sessions(db: AsyncSession, user_id: int, reason: str) -> int:
    """Close every session of a user. Does not commit; returns tokens revoked."""
    result = await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at_ms.is_(None),
        )
        .values(revoked_at_ms=now_ms(), revoked_reason=reason)
    )
    return int(getattr(result, "rowcount", 0) or 0)


def session_id_of(token: str) -> tuple[str, int] | None:
    """Session id and user id of a valid refresh token, or None."""
    try:
        payload = decode_refresh_token(token)
        return str(payload["sid"]), int(str(payload["sub"]))
    except jwt.PyJWTError, KeyError, ValueError:
        return None


async def open_session_counts(db: AsyncSession, user_ids: list[int]) -> dict[int, int]:
    """Number of open, unexpired sessions per user."""
    if not user_ids:
        return {}
    rows = await db.execute(
        select(RefreshToken.user_id, func.count(func.distinct(RefreshToken.family_id)))
        .where(
            RefreshToken.user_id.in_(user_ids),
            RefreshToken.revoked_at_ms.is_(None),
            RefreshToken.used_at_ms.is_(None),
            RefreshToken.expires_at_ms > now_ms(),
        )
        .group_by(RefreshToken.user_id)
    )
    return {int(user_id): int(count) for user_id, count in rows.all()}
