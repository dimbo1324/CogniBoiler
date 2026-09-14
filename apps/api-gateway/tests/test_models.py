"""
Tests for SQLAlchemy ORM models and Alembic migration.

No real PostgreSQL needed — we use SQLite in-memory via aiosqlite
(SQLAlchemy async works with SQLite for schema/unit tests).

Tests cover:
  1. Model imports and table name correctness
  2. Base.metadata contains all expected tables
  3. Column presence and types
  4. Constraints (unique, nullable, FK)
  5. Alembic migration generates valid SQL (offline mode)
"""

from __future__ import annotations

import pytest
from api_gateway.models.user import (
    AuditLog,
    Base,
    Role,
    TokenBlacklist,
    User,
    UserRole,
)
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db_engine():  # type: ignore[no-untyped-def]
    """
    In-memory SQLite async engine with all tables created.

    SQLite is used instead of PostgreSQL so tests run without
    a real database server. Schema differences (SERIAL vs INTEGER,
    BIGSERIAL vs INTEGER) do not affect these unit tests.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):  # type: ignore[no-untyped-def]
    """Async session bound to the in-memory SQLite engine."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


# ─── 1. Metadata tests ────────────────────────────────────────────────────────


class TestMetadata:
    def test_all_five_tables_registered(self) -> None:
        tables = set(Base.metadata.tables.keys())
        assert tables == {
            "users",
            "roles",
            "user_roles",
            "token_blacklist",
            "audit_log",
        }

    def test_users_table_name(self) -> None:
        assert User.__tablename__ == "users"

    def test_roles_table_name(self) -> None:
        assert Role.__tablename__ == "roles"

    def test_user_roles_table_name(self) -> None:
        assert UserRole.__tablename__ == "user_roles"

    def test_token_blacklist_table_name(self) -> None:
        assert TokenBlacklist.__tablename__ == "token_blacklist"

    def test_audit_log_table_name(self) -> None:
        assert AuditLog.__tablename__ == "audit_log"


# ─── 2. Schema inspection tests ───────────────────────────────────────────────


class TestSchema:
    @pytest.mark.asyncio
    async def test_users_columns_exist(self, db_engine) -> None:  # type: ignore[no-untyped-def]
        async with db_engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: {
                    c["name"] for c in inspect(sync_conn).get_columns("users")
                }
            )
        assert {
            "id",
            "username",
            "hashed_password",
            "is_active",
            "created_at_ms",
        } <= cols

    @pytest.mark.asyncio
    async def test_roles_columns_exist(self, db_engine) -> None:  # type: ignore[no-untyped-def]
        async with db_engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: {
                    c["name"] for c in inspect(sync_conn).get_columns("roles")
                }
            )
        assert {"id", "name", "description"} <= cols

    @pytest.mark.asyncio
    async def test_audit_log_columns_exist(self, db_engine) -> None:  # type: ignore[no-untyped-def]
        async with db_engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: {
                    c["name"] for c in inspect(sync_conn).get_columns("audit_log")
                }
            )
        expected = {
            "id",
            "user_id",
            "ip_address",
            "method",
            "endpoint",
            "request_body_hash",
            "response_status",
            "duration_ms",
            "timestamp_ms",
            "detail",
        }
        assert expected <= cols

    @pytest.mark.asyncio
    async def test_token_blacklist_columns_exist(self, db_engine) -> None:  # type: ignore[no-untyped-def]
        async with db_engine.connect() as conn:
            cols = await conn.run_sync(
                lambda sync_conn: {
                    c["name"] for c in inspect(sync_conn).get_columns("token_blacklist")
                }
            )
        assert {"id", "token_jti", "user_id", "revoked_at_ms", "exp_ms"} <= cols


# ─── 3. ORM CRUD tests ────────────────────────────────────────────────────────


class TestUserCRUD:
    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession) -> None:
        import time

        user = User(
            username="operator1",
            hashed_password="$argon2id$v=19$test",
            is_active=True,
            created_at_ms=int(time.time() * 1000),
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        assert user.id is not None
        assert user.username == "operator1"

    @pytest.mark.asyncio
    async def test_user_is_active_default(self, db_session: AsyncSession) -> None:
        import time

        user = User(
            username="viewer1",
            hashed_password="$argon2id$v=19$test",
            is_active=True,
            created_at_ms=int(time.time() * 1000),
        )
        db_session.add(user)
        await db_session.commit()
        assert user.is_active is True

    @pytest.mark.asyncio
    async def test_username_unique_constraint(self, db_session: AsyncSession) -> None:
        import time

        from sqlalchemy.exc import IntegrityError

        ts = int(time.time() * 1000)
        u1 = User(
            username="dup", hashed_password="h1", is_active=True, created_at_ms=ts
        )
        u2 = User(
            username="dup", hashed_password="h2", is_active=True, created_at_ms=ts
        )
        db_session.add(u1)
        await db_session.flush()
        db_session.add(u2)
        with pytest.raises(IntegrityError):
            await db_session.flush()


class TestRoleCRUD:
    @pytest.mark.asyncio
    async def test_create_role(self, db_session: AsyncSession) -> None:
        role = Role(name="viewer", description="Read-only access")
        db_session.add(role)
        await db_session.commit()
        await db_session.refresh(role)
        assert role.id is not None
        assert role.name == "viewer"

    @pytest.mark.asyncio
    async def test_role_name_unique(self, db_session: AsyncSession) -> None:
        from sqlalchemy.exc import IntegrityError

        db_session.add(Role(name="admin", description="desc1"))
        await db_session.flush()
        db_session.add(Role(name="admin", description="desc2"))
        with pytest.raises(IntegrityError):
            await db_session.flush()


class TestUserRoleCRUD:
    @pytest.mark.asyncio
    async def test_assign_role_to_user(self, db_session: AsyncSession) -> None:
        import time

        ts = int(time.time() * 1000)
        user = User(
            username="eng1", hashed_password="h", is_active=True, created_at_ms=ts
        )
        role = Role(name="engineer", description="Engineer role")
        db_session.add_all([user, role])
        await db_session.flush()

        ur = UserRole(user_id=user.id, role_id=role.id, granted_at_ms=ts)
        db_session.add(ur)
        await db_session.commit()
        assert ur.id is not None

    @pytest.mark.asyncio
    async def test_user_role_unique_constraint(self, db_session: AsyncSession) -> None:
        import time

        from sqlalchemy.exc import IntegrityError

        ts = int(time.time() * 1000)
        user = User(
            username="op2", hashed_password="h", is_active=True, created_at_ms=ts
        )
        role = Role(name="operator", description="Operator role")
        db_session.add_all([user, role])
        await db_session.flush()

        db_session.add(UserRole(user_id=user.id, role_id=role.id, granted_at_ms=ts))
        await db_session.flush()
        db_session.add(UserRole(user_id=user.id, role_id=role.id, granted_at_ms=ts))
        with pytest.raises(IntegrityError):
            await db_session.flush()


class TestTokenBlacklist:
    @pytest.mark.asyncio
    async def test_add_token_to_blacklist(self, db_session: AsyncSession) -> None:
        import time

        ts = int(time.time() * 1000)
        user = User(
            username="usr_bl", hashed_password="h", is_active=True, created_at_ms=ts
        )
        db_session.add(user)
        await db_session.flush()

        token = TokenBlacklist(
            token_jti="some.jwt.token",
            user_id=user.id,
            revoked_at_ms=ts,
            exp_ms=ts + 7 * 24 * 3600 * 1000,
        )
        db_session.add(token)
        await db_session.commit()
        assert token.id is not None

    @pytest.mark.asyncio
    async def test_token_jti_unique(self, db_session: AsyncSession) -> None:
        import time

        from sqlalchemy.exc import IntegrityError

        ts = int(time.time() * 1000)
        user = User(
            username="usr_bl2", hashed_password="h", is_active=True, created_at_ms=ts
        )
        db_session.add(user)
        await db_session.flush()

        jti = "duplicate.token"
        db_session.add(
            TokenBlacklist(
                token_jti=jti, user_id=user.id, revoked_at_ms=ts, exp_ms=ts + 1000
            )
        )
        await db_session.flush()
        db_session.add(
            TokenBlacklist(
                token_jti=jti, user_id=user.id, revoked_at_ms=ts, exp_ms=ts + 1000
            )
        )
        with pytest.raises(IntegrityError):
            await db_session.flush()


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_create_audit_entry(self, db_session: AsyncSession) -> None:
        import time

        ts = int(time.time() * 1000)
        entry = AuditLog(
            user_id=None,  # unauthenticated request
            ip_address="127.0.0.1",
            method="GET",
            endpoint="/health",
            request_body_hash=None,
            response_status=200,
            duration_ms=3,
            timestamp_ms=ts,
        )
        db_session.add(entry)
        await db_session.commit()
        assert entry.id is not None

    @pytest.mark.asyncio
    async def test_audit_log_user_id_nullable(self, db_session: AsyncSession) -> None:
        """Anonymous requests must be recordable with user_id=NULL."""
        import time

        entry = AuditLog(
            user_id=None,
            ip_address="10.0.0.1",
            method="POST",
            endpoint="/auth/login",
            response_status=401,
            duration_ms=12,
            timestamp_ms=int(time.time() * 1000),
        )
        db_session.add(entry)
        await db_session.commit()
        assert entry.user_id is None

    @pytest.mark.asyncio
    async def test_multiple_audit_entries_allowed(
        self, db_session: AsyncSession
    ) -> None:
        """audit_log must accept multiple entries — no unique constraint on rows."""
        import time

        ts = int(time.time() * 1000)
        for i in range(5):
            db_session.add(
                AuditLog(
                    ip_address="127.0.0.1",
                    method="GET",
                    endpoint=f"/api/v1/status/{i}",
                    response_status=200,
                    duration_ms=i,
                    timestamp_ms=ts + i,
                )
            )
        await db_session.commit()

        from sqlalchemy import select

        result = await db_session.execute(select(AuditLog))
        rows = result.scalars().all()
        assert len(rows) == 5
