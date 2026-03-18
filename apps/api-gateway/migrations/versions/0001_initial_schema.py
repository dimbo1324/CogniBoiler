"""Initial schema: users, roles, user_roles, token_blacklist, audit_log

Revision ID: 0001
Revises:
Create Date: 2026-03-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── users ──────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "username", sa.String(64), nullable=False, comment="Unique login name"
        ),
        sa.Column(
            "hashed_password",
            sa.String(256),
            nullable=False,
            comment="Argon2id encoded hash — never plain text",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            comment="False = soft-deleted, cannot log in",
        ),
        sa.Column(
            "created_at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="Account creation time [UTC epoch ms]",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    # ── roles ──────────────────────────────────────────────────────────────
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "name",
            sa.String(32),
            nullable=False,
            comment="Role name: viewer | operator | engineer | admin",
        ),
        sa.Column(
            "description",
            sa.String(256),
            nullable=False,
            comment="Human-readable description of this role",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # ── user_roles ─────────────────────────────────────────────────────────
    op.create_table(
        "user_roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column(
            "granted_at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="When this role was granted [UTC epoch ms]",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "role_id", name="uq_user_role"),
    )
    op.create_index("ix_user_roles_user_id", "user_roles", ["user_id"])
    op.create_index("ix_user_roles_role_id", "user_roles", ["role_id"])

    # ── token_blacklist ────────────────────────────────────────────────────
    op.create_table(
        "token_blacklist",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "token_jti",
            sa.String(256),
            nullable=False,
            comment="Full refresh token string (or JWT jti claim)",
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
            comment="Owner of the invalidated token",
        ),
        sa.Column(
            "revoked_at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="When the token was revoked [UTC epoch ms]",
        ),
        sa.Column(
            "exp_ms",
            sa.BigInteger(),
            nullable=False,
            comment="Token natural expiry [UTC epoch ms] — for cleanup jobs",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_jti"),
    )
    op.create_index(
        "ix_token_blacklist_token_jti", "token_blacklist", ["token_jti"], unique=True
    )
    op.create_index("ix_token_blacklist_user_id", "token_blacklist", ["user_id"])

    # ── audit_log ──────────────────────────────────────────────────────────
    # INSERT-only in production — UPDATE/DELETE blocked at DB user level
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=True,
            comment="NULL for unauthenticated requests",
        ),
        sa.Column(
            "ip_address",
            sa.String(45),
            nullable=False,
            comment="IPv4 or IPv6 client address",
        ),
        sa.Column(
            "method",
            sa.String(10),
            nullable=False,
            comment="HTTP method: GET, POST, PUT, DELETE, ...",
        ),
        sa.Column(
            "endpoint",
            sa.String(256),
            nullable=False,
            comment="Request path, e.g. /api/v1/commands/valve",
        ),
        sa.Column(
            "request_body_hash",
            sa.String(64),
            nullable=True,
            comment="SHA-256 hex digest of request body, NULL if no body",
        ),
        sa.Column(
            "response_status",
            sa.Integer(),
            nullable=False,
            comment="HTTP response status code",
        ),
        sa.Column(
            "duration_ms",
            sa.Integer(),
            nullable=False,
            comment="Request processing time [ms]",
        ),
        sa.Column(
            "timestamp_ms",
            sa.BigInteger(),
            nullable=False,
            comment="Request received time [UTC epoch ms]",
        ),
        sa.Column(
            "detail",
            sa.Text(),
            nullable=True,
            comment="Optional extra context (error message, command parameters)",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_timestamp_ms", "audit_log", ["timestamp_ms"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])

    # ── Seed initial roles ─────────────────────────────────────────────────
    op.execute(
        "INSERT INTO roles (name, description) VALUES "
        "('viewer',   'Read-only access to status and history'), "
        "('operator', 'Valve commands within safe limits'), "
        "('engineer', 'PID setpoint changes and model parameters'), "
        "('admin',    'User management and security configuration')"
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("token_blacklist")
    op.drop_table("user_roles")
    op.drop_table("roles")
    op.drop_table("users")
