"""
Role-Based Access Control (RBAC) via FastAPI dependency injection.

Role hierarchy (least -> most privileged):
    viewer   -> read-only access to status and history
    operator -> viewer + valve commands within safe limits
    engineer -> operator + PID setpoint changes, model parameters
    admin    -> engineer + user management, security configuration

How it works:
    Each router endpoint declares a dependency:
        @router.post("/commands/valve")
        async def send_valve_command(
            _: TokenData = Depends(require_role("operator")),
        ):
            ...

    FastAPI automatically resolves the dependency chain on every request:
        1. extract_bearer_token() pulls the raw JWT from Authorization header
        2. decode_access_token()  verifies RS256 signature and expiry
        3. require_role()         checks the role claim against the hierarchy
        4. Returns the decoded payload (available to the endpoint if needed)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api_gateway.auth.jwt_handler import TokenData, decode_access_token

# ─── Role hierarchy ───────────────────────────────────────────────────────────

# Roles ordered from least to most privileged.
_ROLE_HIERARCHY: list[str] = ["viewer", "operator", "engineer", "admin"]


def _role_level(role: str) -> int:
    """
    Return the numeric privilege level of a role.

    Unknown roles return -1, effectively denying all access.

    Args:
        role: Role string from the JWT payload.

    Returns:
        Integer level: 0 (viewer) … 3 (admin), or -1 if unknown.
    """
    try:
        return _ROLE_HIERARCHY.index(role)
    except ValueError:
        return -1


# ─── Bearer token extractor ───────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=True)


async def extract_bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
) -> str:
    """
    FastAPI dependency: extract the raw JWT string from the request.

    Returns:
        Raw JWT string.
    """
    return credentials.credentials  # type: ignore[no-any-return]


# ─── Current user dependency ─────────────────────────────────────────────────


async def get_current_user(
    token: Annotated[str, Depends(extract_bearer_token)],
) -> TokenData:
    """
    FastAPI dependency: verify the JWT and return the decoded payload.

    Raises:
        HTTPException 401: Token is invalid or expired.
    """
    try:
        return decode_access_token(token)
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─── Role-based access guard ──────────────────────────────────────────────────


def require_role(minimum_role: str) -> Callable[..., Awaitable[TokenData]]:
    """
    FastAPI dependency factory: enforce a minimum role requirement.

    Returns a dependency function that checks the authenticated user's
    role against the required minimum. Uses the role hierarchy so that
    higher-privileged users always pass lower-privilege checks.

    Usage:
        @router.post("/commands/valve")
        async def send_command(
            payload: TokenData = Depends(require_role("operator")),
        ):
            user_id = payload["sub"]

    Args:
        minimum_role: Minimum role required ("viewer", "operator",
                      "engineer", or "admin").

    Returns:
        An async FastAPI dependency function.

    Raises:
        HTTPException 403: User's role is below the required minimum.
        HTTPException 401: Token is invalid (propagated from get_current_user).
    """
    required_level = _role_level(minimum_role)

    async def _dependency(
        payload: Annotated[TokenData, Depends(get_current_user)],
    ) -> TokenData:
        user_role = str(payload.get("role", ""))
        if _role_level(user_role) < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient permissions. "
                    f"Required: {minimum_role}, your role: {user_role}."
                ),
            )
        return payload

    return _dependency
