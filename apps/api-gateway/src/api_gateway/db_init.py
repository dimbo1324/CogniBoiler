"""Development/bootstrap DB initialization for local core stack runs."""

from __future__ import annotations

import time

from alert_manager.models import Base as AlertBase
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.auth.password import hash_password
from api_gateway.dependencies import AsyncSessionLocal, engine
from api_gateway.models.user import Base, Role, User, UserRole


async def ensure_schema_and_seed_defaults() -> None:
    """
    Create tables and seed roles/default users for local development.

    This is intentionally lightweight and meant for docker-compose/local runs.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(AlertBase.metadata.create_all)

    async with AsyncSessionLocal() as session:
        role_rows = (
            (
                await session.execute(
                    select(Role).where(
                        Role.name.in_(["viewer", "operator", "engineer", "admin"])
                    )
                )
            )
            .scalars()
            .all()
        )
        existing_roles = {role.name: role for role in role_rows}
        for name, description in (
            ("viewer", "Read-only access to status and history"),
            ("operator", "Valve commands within safe limits"),
            ("engineer", "PID setpoint changes and model parameters"),
            ("admin", "User management and security configuration"),
        ):
            if name not in existing_roles:
                role = Role(name=name, description=description)
                session.add(role)
                await session.flush()
                existing_roles[name] = role

        await _ensure_default_user(
            session, existing_roles["admin"], "admin", "cogniboiler-admin"
        )
        await _ensure_default_user(
            session, existing_roles["operator"], "operator", "cogniboiler-operator"
        )
        await _ensure_default_user(
            session, existing_roles["engineer"], "engineer", "cogniboiler-engineer"
        )
        await _ensure_default_user(
            session, existing_roles["viewer"], "viewer", "cogniboiler-viewer"
        )
        await session.commit()


async def _ensure_default_user(
    session: AsyncSession,
    role: Role,
    username: str,
    password: str,
) -> None:
    """Create a default user and role assignment if it is missing."""
    user = await session.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(
            username=username,
            hashed_password=hash_password(password),
            is_active=True,
            created_at_ms=int(time.time() * 1000),
        )
        session.add(user)
        await session.flush()

    assignment = await session.scalar(
        select(UserRole).where(
            UserRole.user_id == user.id,
            UserRole.role_id == role.id,
        )
    )
    if assignment is None:
        session.add(
            UserRole(
                user_id=user.id,
                role_id=role.id,
                granted_at_ms=int(time.time() * 1000),
            )
        )
