"""
Account administration and self-service password changes.

Rules that protect the installation from locking itself out:
  - an administrator cannot change their own role or block themselves;
  - the last active administrator cannot be demoted or blocked.
Any change to a role, the active flag or a password closes the user's sessions, so the
change takes effect immediately rather than when tokens expire.

Argon2 hashing is deliberately slow; it runs in a worker thread so the event loop keeps
serving telemetry.
"""

from __future__ import annotations

import asyncio
import time

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.auth.identity import Account, CurrentUser, effective_role, load_account
from api_gateway.auth.password import hash_password, verify_password
from api_gateway.auth.sessions import open_session_counts, revoke_user_sessions
from api_gateway.models.user import Role, User, UserRole
from api_gateway.problems import ProblemError
from api_gateway.schemas.users import (
    UserCreateRequest,
    UserPageResponse,
    UserResponse,
    UserUpdateRequest,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def hash_password_async(plain: str) -> str:
    return await asyncio.to_thread(hash_password, plain)


async def verify_password_async(plain: str, hashed: str) -> bool:
    return await asyncio.to_thread(verify_password, plain, hashed)


def check_password_policy(username: str, password: str) -> None:
    """Length is validated by the schema; here, what length cannot express."""
    folded = password.casefold()
    if username.casefold() in folded:
        raise ProblemError(
            422,
            "users.password_weak",
            "The password must not contain the username.",
        )
    if len(set(password)) < 5:
        raise ProblemError(
            422,
            "users.password_weak",
            "The password must use at least five different characters.",
        )


async def _role_row(db: AsyncSession, name: str) -> Role:
    role = await db.scalar(select(Role).where(Role.name == name))
    if role is None:
        raise ProblemError(
            409, "users.role_missing", f"The role {name} is not provisioned."
        )
    return role


async def _account_or_404(db: AsyncSession, user_id: int) -> Account:
    account = await load_account(db, user_id=user_id)
    if account is None:
        raise ProblemError(404, "users.not_found", f"User {user_id} does not exist.")
    return account


async def _active_admin_count(db: AsyncSession) -> int:
    count = await db.scalar(
        select(func.count(func.distinct(User.id)))
        .join(UserRole, UserRole.user_id == User.id)
        .join(Role, Role.id == UserRole.role_id)
        .where(User.is_active.is_(True), Role.name == "admin")
    )
    return int(count or 0)


def _response(account: Account, open_sessions: int) -> UserResponse:
    return UserResponse.model_validate(
        {
            "id": account.id,
            "username": account.username,
            "role": account.role or None,
            "is_active": account.is_active,
            "created_at_ms": account.created_at_ms,
            "last_login_at_ms": account.last_login_at_ms,
            "open_sessions": open_sessions,
        }
    )


async def list_users(
    db: AsyncSession, *, active: bool | None, limit: int, offset: int
) -> UserPageResponse:
    filters = [] if active is None else [User.is_active.is_(active)]
    total = await db.scalar(select(func.count(User.id)).where(*filters))
    users = (
        (
            await db.execute(
                select(User)
                .where(*filters)
                .order_by(User.username)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    ids = [user.id for user in users]
    role_rows = (
        await db.execute(
            select(UserRole.user_id, Role.name)
            .join(Role, Role.id == UserRole.role_id)
            .where(UserRole.user_id.in_(ids))
        )
    ).all()
    roles: dict[int, list[str]] = {}
    for user_id, name in role_rows:
        roles.setdefault(int(user_id), []).append(str(name))
    sessions = await open_session_counts(db, ids)
    items = [
        _response(
            Account(
                id=user.id,
                username=user.username,
                hashed_password="",
                is_active=user.is_active,
                role=effective_role(roles.get(user.id, [])),
                created_at_ms=user.created_at_ms,
                last_login_at_ms=user.last_login_at_ms,
            ),
            sessions.get(user.id, 0),
        )
        for user in users
    ]
    return UserPageResponse(
        items=items, total=int(total or 0), limit=limit, offset=offset
    )


async def get_user(db: AsyncSession, user_id: int) -> UserResponse:
    account = await _account_or_404(db, user_id)
    sessions = await open_session_counts(db, [user_id])
    return _response(account, sessions.get(user_id, 0))


async def create_user(db: AsyncSession, request: UserCreateRequest) -> UserResponse:
    check_password_policy(request.username, request.password)
    role = await _role_row(db, request.role)
    exists = await db.scalar(
        select(User.id).where(func.lower(User.username) == request.username.lower())
    )
    if exists is not None:
        raise ProblemError(
            409, "users.username_taken", "An account with this username exists."
        )
    now = _now_ms()
    user = User(
        username=request.username,
        hashed_password=await hash_password_async(request.password),
        is_active=True,
        created_at_ms=now,
    )
    db.add(user)
    try:
        await db.flush()
        db.add(UserRole(user_id=user.id, role_id=role.id, granted_at_ms=now))
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise ProblemError(
            409, "users.username_taken", "An account with this username exists."
        ) from exc
    return await get_user(db, user.id)


async def update_user(
    db: AsyncSession, actor: CurrentUser, user_id: int, request: UserUpdateRequest
) -> UserResponse:
    account = await _account_or_404(db, user_id)
    role_changes = request.role is not None and request.role != account.role
    activity_changes = (
        request.is_active is not None and request.is_active != account.is_active
    )
    if not role_changes and not activity_changes:
        return await get_user(db, user_id)

    if user_id == actor.id:
        raise ProblemError(
            409,
            "users.self_change",
            "You cannot change your own role or block your own account.",
        )
    loses_admin = (
        account.role == "admin"
        and account.is_active
        and ((role_changes and request.role != "admin") or request.is_active is False)
    )
    if loses_admin and await _active_admin_count(db) <= 1:
        raise ProblemError(
            409,
            "users.last_admin",
            "The last active administrator cannot be demoted or blocked.",
        )

    user = await db.get(User, user_id)
    if user is None:
        raise ProblemError(404, "users.not_found", f"User {user_id} does not exist.")
    if role_changes and request.role is not None:
        role = await _role_row(db, request.role)
        await db.execute(delete(UserRole).where(UserRole.user_id == user_id))
        db.add(UserRole(user_id=user_id, role_id=role.id, granted_at_ms=_now_ms()))
    if activity_changes and request.is_active is not None:
        user.is_active = request.is_active
    reason = "blocked" if request.is_active is False else "role_change"
    await revoke_user_sessions(db, user_id, reason)
    await db.commit()
    return await get_user(db, user_id)


async def reset_password(db: AsyncSession, user_id: int, new_password: str) -> None:
    account = await _account_or_404(db, user_id)
    check_password_policy(account.username, new_password)
    user = await db.get(User, user_id)
    if user is None:
        raise ProblemError(404, "users.not_found", f"User {user_id} does not exist.")
    user.hashed_password = await hash_password_async(new_password)
    await revoke_user_sessions(db, user_id, "admin")
    await db.commit()


async def change_own_password(
    db: AsyncSession, actor: CurrentUser, current_password: str, new_password: str
) -> Account:
    """Verify the current password and set a new one; closes every session."""
    account = await _account_or_404(db, actor.id)
    if not await verify_password_async(current_password, account.hashed_password):
        raise ProblemError(
            403, "auth.password_mismatch", "The current password is not correct."
        )
    if current_password == new_password:
        raise ProblemError(
            422,
            "users.password_unchanged",
            "The new password must differ from the current one.",
        )
    check_password_policy(account.username, new_password)
    user = await db.get(User, actor.id)
    if user is None:
        raise ProblemError(404, "users.not_found", f"User {actor.id} does not exist.")
    user.hashed_password = await hash_password_async(new_password)
    await revoke_user_sessions(db, actor.id, "password_change")
    await db.commit()
    return account


async def revoke_sessions(db: AsyncSession, user_id: int) -> int:
    await _account_or_404(db, user_id)
    revoked = await revoke_user_sessions(db, user_id, "admin")
    await db.commit()
    return revoked
