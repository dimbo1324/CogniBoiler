"""
SQLAlchemy ORM models for the CogniBoiler API Gateway.

Tables:
    users           — registered operator accounts
    roles           — RBAC role definitions
    user_roles      — many-to-many: users ↔ roles
    token_blacklist — invalidated refresh tokens (logout)
    audit_log       — immutable record of every API call

Immutability of audit_log:
    The audit_log table is INSERT-only by design. In production,
    a dedicated PostgreSQL user with GRANT INSERT (no UPDATE/DELETE)
    is used. This is enforced at the DB level, not in Python.

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


class Base(DeclarativeBase):  # type: ignore[misc]
    """Shared declarative base for all ORM models."""

    pass


# ─── users ────────────────────────────────────────────────────────────────────


class User(Base):
    """
    Registered user account.

    Passwords are stored as Argon2id hashes — never plain text.
    The is_active flag allows soft-deletion without removing audit history.
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
        comment="False = soft-deleted, cannot log in",
    )
    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="Account creation time [UTC epoch ms]",
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
    The hierarchy is enforced by rbac.py, not by the DB schema.
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

    One user can have multiple roles, though in practice each user
    has exactly one role in the current implementation.
    The unique constraint prevents duplicate assignments.
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


# ─── token_blacklist ──────────────────────────────────────────────────────────


class TokenBlacklist(Base):
    """
    Invalidated refresh tokens (logout / rotation).

    When a user logs out, their refresh token is inserted here.
    The auth router checks this table before issuing new access tokens.

    Old entries (past exp_ms) can be purged by a scheduled job —
    they serve no purpose after natural expiry.
    """

    __tablename__ = "token_blacklist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_jti: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        unique=True,
        index=True,
        comment="Full refresh token string (or JWT jti claim)",
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner of the invalidated token",
    )
    revoked_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="When the token was revoked [UTC epoch ms]",
    )
    exp_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="Token natural expiry [UTC epoch ms] — for cleanup jobs",
    )

    def __repr__(self) -> str:
        return f"<TokenBlacklist id={self.id} user_id={self.user_id}>"


# ─── audit_log ────────────────────────────────────────────────────────────────


class AuditLog(Base):
    """
    Immutable audit trail of every API call.

    Every request processed by the API Gateway is recorded here.
    In production, the DB user writing to this table has INSERT-only
    privileges — UPDATE and DELETE are physically impossible.

    Fields:
        user_id:            Who made the request (NULL for anonymous).
        ip_address:         Client IP address.
        method:             HTTP method (GET, POST, etc.).
        endpoint:           Request path (/api/v1/status).
        request_body_hash:  SHA-256 of request body (not stored in full).
        response_status:    HTTP status code returned.
        duration_ms:        Request processing time [ms].
        timestamp_ms:       When the request was received [UTC epoch ms].
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
    ip_address: Mapped[str] = mapped_column(
        String(45),
        nullable=False,
        comment="IPv4 or IPv6 client address",
    )
    method: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="HTTP method: GET, POST, PUT, DELETE, ...",
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
        comment="Optional extra context (error message, command parameters)",
    )

    user: Mapped[User | None] = relationship("User", back_populates="audit_logs")

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} method={self.method!r} "
            f"endpoint={self.endpoint!r} status={self.response_status}>"
        )
