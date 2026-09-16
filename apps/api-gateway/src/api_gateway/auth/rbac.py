"""
Role-Based Access Control (RBAC) via FastAPI dependency injection.

Role hierarchy (least -> most privileged):
    viewer   -> read-only access to status, history, alarms
    operator -> viewer + load, mode, valve commands, alarm acknowledgement
    engineer -> operator + setpoints, E-Stop reset, simulation control
    admin    -> engineer + users and the audit log

Each endpoint declares the minimum role through one of the annotated aliases:

    @router.post("/commands/load")
    async def set_load(user: OperatorUser) -> ...:
        ...

FastAPI resolves the chain on every request: the bearer token is read, resolved to an
active user in an open session (database), and the user's current role is compared with
the requirement. The user is also handed to the audit middleware.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_gateway.audit import AuditActor, set_audit_actor
from api_gateway.auth.identity import (
    AuthenticationError,
    CurrentUser,
    resolve_access_token,
    role_level,
)
from api_gateway.dependencies import DbSession
from api_gateway.problems import ProblemError

_bearer_scheme = HTTPBearer(auto_error=False)

_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _role_level(role: str) -> int:
    """Numeric privilege level of a role; -1 for unknown roles."""
    return role_level(role)


async def get_current_user(
    request: Request,
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
) -> CurrentUser:
    """
    FastAPI dependency: the active user behind the bearer token.

    Raises:
        ProblemError 401: no token, an invalid or expired token, a closed session or a
            blocked account.
    """
    if credentials is None:
        raise ProblemError(
            401,
            "auth.token_missing",
            "Authentication is required.",
            headers=_BEARER_CHALLENGE,
        )
    try:
        user = await resolve_access_token(db, credentials.credentials)
    except AuthenticationError as exc:
        raise ProblemError(
            401, exc.code, exc.detail, headers=_BEARER_CHALLENGE
        ) from exc
    set_audit_actor(request, AuditActor(user.id, user.username, user.role))
    return user


def require_role(minimum_role: str) -> Callable[..., Awaitable[CurrentUser]]:
    """
    FastAPI dependency factory: the current user, if their role is at least minimum_role.

    Raises:
        ProblemError 403: the user's role is below the requirement.
        ProblemError 401: propagated from get_current_user.
    """

    async def _dependency(
        user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if not user.at_least(minimum_role):
            raise ProblemError(
                403,
                "auth.forbidden",
                f"This action requires the {minimum_role} role or higher.",
                extra={"required_role": minimum_role, "role": user.role},
            )
        return user

    return _dependency


AuthenticatedUser = Annotated[CurrentUser, Depends(get_current_user)]
ViewerUser = Annotated[CurrentUser, Depends(require_role("viewer"))]
OperatorUser = Annotated[CurrentUser, Depends(require_role("operator"))]
EngineerUser = Annotated[CurrentUser, Depends(require_role("engineer"))]
AdminUser = Annotated[CurrentUser, Depends(require_role("admin"))]
