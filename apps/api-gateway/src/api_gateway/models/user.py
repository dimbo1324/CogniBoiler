"""
SQLAlchemy ORM models for the CogniBoiler API Gateway.

Tables:
    users          — registered accounts; blocked, never deleted
    roles          — RBAC role definitions
    user_roles     — many-to-many: users ↔ roles
    refresh_tokens — sign-in sessions: one family of rotated refresh tokens per sign-in
    audit_log      — append-only record of mutations, sign-ins and refusals
    scenario_runs  — who loaded which scenario and injected or cleared which fault

The append-only rule of audit_log is enforced by database triggers created in the
migration chain (revision 0003), not by this module.

All timestamps are stored as UTC epoch milliseconds (int) for
consistency with the protobuf schema and InfluxDB timestamps.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ─── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ─── users ────────────────────────────────────────────────────────────────────


class User(Base):
    """
    Registered user account.

    Passwords are stored as Argon2id hashes — never plain text. An account is blocked
    with is_active and never deleted, so the audit trail keeps pointing at it.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    username: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique login name",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Argon2id encoded hash — never plain text",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="False = blocked, cannot sign in",
    )
    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="Account creation time [UTC epoch ms]",
    )
    last_login_at_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Latest successful sign-in [UTC epoch ms]",
    )

    # Relationships
    user_roles: Mapped[list[UserRole]] = relationship(
        "UserRole", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list[AuditLog]] = relationship("AuditLog", back_populates="user")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


# ─── roles ────────────────────────────────────────────────────────────────────


class Role(Base):
    """
    RBAC role definition.

    Pre-seeded roles: viewer, operator, engineer, admin.
    The hierarchy is enforced by the auth package, not by the DB schema.
    """

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
        comment="Role name: viewer | operator | engineer | admin",
    )
    description: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        default="",
        comment="Human-readable description of this role",
    )

    user_roles: Mapped[list[UserRole]] = relationship("UserRole", back_populates="role")

    def __repr__(self) -> str:
        return f"<Role id={self.id} name={self.name!r}>"


# ─── user_roles ───────────────────────────────────────────────────────────────


class UserRole(Base):
    """
    Many-to-many association between users and roles.

    The API keeps exactly one role per user; if several rows exist, the most
    privileged one is effective. The unique constraint prevents duplicates.
    """

    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role_id", name="uq_user_role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    granted_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="When this role was granted [UTC epoch ms]",
    )

    user: Mapped[User] = relationship("User", back_populates="user_roles")
    role: Mapped[Role] = relationship("Role", back_populates="user_roles")

    def __repr__(self) -> str:
        return f"<UserRole user_id={self.user_id} role_id={self.role_id}>"


# ─── refresh_tokens ───────────────────────────────────────────────────────────


class RefreshToken(Base):
    """
    One issued refresh token of a sign-in session.

    Every sign-in opens a family; each refresh marks the presented token used and
    issues its successor in the same family with the same absolute expiry. Access
    tokens carry the family id, so revoking a family ends the session at once.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    jti: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, index=True, comment="JWT id"
    )
    family_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True, comment="Session id shared by rotations"
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    issued_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="Absolute session expiry [UTC epoch ms]"
    )
    used_at_ms: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Exchanged for its successor [UTC epoch ms]"
    )
    replaced_by_jti: Mapped[str | None] = mapped_column(String(36), nullable=True)
    revoked_at_ms: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Session closed [UTC epoch ms]"
    )
    revoked_reason: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="logout | reuse | password_change | role_change | blocked | admin",
    )
    client_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(256), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<RefreshToken id={self.id} family={self.family_id} "
            f"user_id={self.user_id}>"
        )


# ─── audit_log ────────────────────────────────────────────────────────────────


class AuditLog(Base):
    """
    Append-only audit trail.

    One row per mutating request, sign-in, sign-out and refused request. The body is
    kept only as a SHA-256 digest; who acted is recorded by id, name and the role they
    held at that moment, and outcome says how the action ended (a PLC refusal is an
    HTTP 200 whose outcome is "refused").
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="NULL for unauthenticated requests",
    )
    username: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Name of the acting user"
    )
    role: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="Role the user held when acting"
    )
    ip_address: Mapped[str] = mapped_column(
        String(45),
        nullable=False,
        comment="IPv4 or IPv6 client address",
    )
    method: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="HTTP method, or WS for WebSocket sign-ins",
    )
    endpoint: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Request path, e.g. /api/v1/commands/valve",
    )
    request_body_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hex digest of request body, NULL if no body",
    )
    response_status: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="HTTP response status code",
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Request processing time [ms]",
    )
    timestamp_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="Request received time [UTC epoch ms]",
    )
    detail: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Query string or the account name of a sign-in attempt",
    )
    outcome: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="How the action ended, e.g. accepted, refused: <reason>",
    )

    user: Mapped[User | None] = relationship("User", back_populates="audit_logs")

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} method={self.method!r} "
            f"endpoint={self.endpoint!r} status={self.response_status}>"
        )


# ─── scenario_runs ────────────────────────────────────────────────────────────


class ScenarioRun(Base):
    """
    A simulation action that changed what the plant is doing: a scenario loaded, a
    fault injected or cleared. Kept next to the audit log so a recorded episode can be
    traced to the person who started it.
    """

    __tablename__ = "scenario_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="scenario | fault_injected | fault_cleared",
    )
    scenario: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="Physics run id after the action"
    )
    fault_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fault_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    severity: Mapped[float | None] = mapped_column(nullable=True)
    simulation_time_s: Mapped[float] = mapped_column(nullable=False)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    at_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, index=True, comment="[UTC epoch ms]"
    )

    def __repr__(self) -> str:
        return f"<ScenarioRun id={self.id} kind={self.kind!r} run_id={self.run_id}>"
