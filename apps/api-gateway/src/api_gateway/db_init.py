"""
Seed data for local development: the four roles and the demo users.

The schema itself comes only from the Alembic chain (the `migrate` job in Compose, or
`alembic upgrade head` from the host); seeding an unmigrated database fails loudly.
"""

from __future__ import annotations

import logging

from cogniboiler_runtime import now_ms
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.auth.password import hash_password
from api_gateway.config import settings
from api_gateway.dependencies import AsyncSessionLocal
from api_gateway.models.user import Role, User, UserRole

logger = logging.getLogger(__name__)

ROLE_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("viewer", "Read-only access to status and history"),
    ("operator", "Valve commands within safe limits"),
    ("engineer", "PID setpoint changes and model parameters"),
    ("admin", "User management and security configuration"),
)


def demo_users() -> tuple[tuple[str, str], ...]:
    """Username and password of each demo user, which is also its role name."""
    return (
        ("admin", settings.demo_admin_password),
        ("engineer", settings.demo_engineer_password),
        ("operator", settings.demo_operator_password),
        ("viewer", settings.demo_viewer_password),
    )


async def seed_roles_and_demo_users() -> None:
    """
    Seed the roles and demo users for local development.

    Passwords come from settings (DEMO_*_PASSWORD); a user whose password is empty is
    not created, so no credential is ever hardcoded here.
    """
    async with AsyncSessionLocal() as session:
        role_names = [name for name, _ in ROLE_DESCRIPTIONS]
        role_rows = (
            (await session.execute(select(Role).where(Role.name.in_(role_names))))
            .scalars()
            .all()
        )
        existing_roles = {role.name: role for role in role_rows}
        for name, description in ROLE_DESCRIPTIONS:
            if name not in existing_roles:
                role = Role(name=name, description=description)
                session.add(role)
                await session.flush()
                existing_roles[name] = role

        for username, password in demo_users():
            if not password:
                logger.warning(
                    "Demo user %r not seeded: its password is empty", username
                )
                continue
            await _ensure_default_user(
                session, existing_roles[username], username, password
            )
        await session.commit()


async def _ensure_default_user(
    session: AsyncSession,
    role: Role,
    username: str,
    password: str,
) -> None:
    """
    Create a missing demo user with its role, or give a role to one that has none.

    A demo user that has any role keeps it: an administrator may have demoted it, and a
    restart must not quietly undo that.
    """
    user = await session.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(
            username=username,
            hashed_password=hash_password(password),
            is_active=True,
            created_at_ms=now_ms(),
        )
        session.add(user)
        await session.flush()
    else:
        any_role = await session.scalar(
            select(UserRole.id).where(UserRole.user_id == user.id).limit(1)
        )
        if any_role is not None:
            return
        logger.warning(
            "Demo user %r had no role; gave it the %r role", username, role.name
        )
    session.add(UserRole(user_id=user.id, role_id=role.id, granted_at_ms=now_ms()))
