"""User administration (admin only). Accounts are blocked, never deleted."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Request, status

from api_gateway import accounts
from api_gateway.audit import set_audit_detail, set_audit_outcome
from api_gateway.auth.rbac import AdminUser
from api_gateway.dependencies import DbSession
from api_gateway.schemas.auth import MessageResponse
from api_gateway.schemas.users import (
    PasswordResetRequest,
    SessionsRevokedResponse,
    UserCreateRequest,
    UserPageResponse,
    UserResponse,
    UserUpdateRequest,
)

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("", response_model=UserPageResponse)
async def list_users(
    db: DbSession,
    _: AdminUser,
    active: bool | None = Query(default=None, description="Only active or blocked."),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> UserPageResponse:
    """Accounts ordered by username, with role and open sessions."""
    return await accounts.list_users(db, active=active, limit=limit, offset=offset)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: Request, body: UserCreateRequest, db: DbSession, _: AdminUser
) -> UserResponse:
    """Create an active account with one role."""
    set_audit_detail(request, f"username={body.username} role={body.role}")
    created = await accounts.create_user(db, body)
    set_audit_outcome(request, f"created user {created.id}")
    return created


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    db: DbSession, _: AdminUser, user_id: int = Path(..., ge=1)
) -> UserResponse:
    return await accounts.get_user(db, user_id)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    request: Request,
    body: UserUpdateRequest,
    db: DbSession,
    admin: AdminUser,
    user_id: int = Path(..., ge=1),
) -> UserResponse:
    """Change the role or block/unblock; the user's sessions are closed."""
    set_audit_detail(
        request, f"user_id={user_id} role={body.role} is_active={body.is_active}"
    )
    updated = await accounts.update_user(db, admin, user_id, body)
    set_audit_outcome(
        request, f"user {user_id}: role={updated.role} active={updated.is_active}"
    )
    return updated


@router.post("/{user_id}/password", response_model=MessageResponse)
async def reset_password(
    request: Request,
    body: PasswordResetRequest,
    db: DbSession,
    _: AdminUser,
    user_id: int = Path(..., ge=1),
) -> MessageResponse:
    """Set a new password for a user; the user's sessions are closed."""
    set_audit_detail(request, f"user_id={user_id}")
    await accounts.reset_password(db, user_id, body.new_password)
    set_audit_outcome(request, f"password of user {user_id} reset")
    return MessageResponse(message="Password reset; the user's sessions are closed.")


@router.post("/{user_id}/revoke-sessions", response_model=SessionsRevokedResponse)
async def revoke_sessions(
    request: Request, db: DbSession, _: AdminUser, user_id: int = Path(..., ge=1)
) -> SessionsRevokedResponse:
    """Sign a user out everywhere."""
    set_audit_detail(request, f"user_id={user_id}")
    revoked = await accounts.revoke_sessions(db, user_id)
    set_audit_outcome(request, f"closed sessions of user {user_id}")
    return SessionsRevokedResponse(user_id=user_id, revoked_tokens=revoked)
