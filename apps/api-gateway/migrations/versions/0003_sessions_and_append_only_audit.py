"""Sessions with rotated refresh tokens, append-only audit, scenario runs

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-16

- refresh_tokens replaces token_blacklist: a sign-in opens a family of refresh tokens,
  each refresh exchanges one token for its successor, and a reused token closes the
  family. Rows of token_blacklist cannot be mapped to families; the table held only
  revoked tokens that expire within a week, so it is dropped with its data.
- users.last_login_at_ms.
- audit_log gains the acting user's name and role and the outcome of the action, and
  becomes append-only: triggers refuse UPDATE, DELETE and (PostgreSQL) TRUNCATE. The
  application connects as the table owner, which could still disable the triggers; a
  separate role without that privilege is hardening work (S11).
- scenario_runs: who loaded a scenario or injected or cleared a fault.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APPEND_ONLY_MESSAGE = "audit_log is append-only"

POSTGRES_APPEND_ONLY = [
    f"""
    CREATE OR REPLACE FUNCTION audit_log_append_only() RETURNS trigger
    LANGUAGE plpgsql AS $$
    BEGIN
        RAISE EXCEPTION '{APPEND_ONLY_MESSAGE}: % refused', TG_OP
            USING ERRCODE = 'insufficient_privilege';
    END
    $$
    """,
    """
    CREATE TRIGGER audit_log_no_update_or_delete
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_append_only()
    """,
    """
    CREATE TRIGGER audit_log_no_truncate
    BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_append_only()
    """,
]

POSTGRES_DROP_APPEND_ONLY = [
    "DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log",
    "DROP TRIGGER IF EXISTS audit_log_no_update_or_delete ON audit_log",
    "DROP FUNCTION IF EXISTS audit_log_append_only()",
]

SQLITE_APPEND_ONLY = [
    f"""
    CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
    BEGIN SELECT RAISE(ABORT, '{APPEND_ONLY_MESSAGE}: UPDATE refused'); END
    """,
    f"""
    CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
    BEGIN SELECT RAISE(ABORT, '{APPEND_ONLY_MESSAGE}: DELETE refused'); END
    """,
]

SQLITE_DROP_APPEND_ONLY = [
    "DROP TRIGGER IF EXISTS audit_log_no_update",
    "DROP TRIGGER IF EXISTS audit_log_no_delete",
]


def _dialect() -> str:
    return op.get_context().dialect.name


def _execute_all(statements: list[str]) -> None:
    for statement in statements:
        op.execute(statement)


def upgrade() -> None:
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("jti", sa.String(36), nullable=False, comment="JWT id"),
        sa.Column(
            "family_id",
            sa.String(36),
            nullable=False,
            comment="Session id shared by rotations",
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("issued_at_ms", sa.BigInteger(), nullable=False),
        sa.Column(
            "expires_at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="Absolute session expiry [UTC epoch ms]",
        ),
        sa.Column(
            "used_at_ms",
            sa.BigInteger(),
            nullable=True,
            comment="Exchanged for its successor [UTC epoch ms]",
        ),
        sa.Column("replaced_by_jti", sa.String(36), nullable=True),
        sa.Column(
            "revoked_at_ms",
            sa.BigInteger(),
            nullable=True,
            comment="Session closed [UTC epoch ms]",
        ),
        sa.Column(
            "revoked_reason",
            sa.String(32),
            nullable=True,
            comment="logout | reuse | password_change | role_change | blocked | admin",
        ),
        sa.Column("client_ip", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(256), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_refresh_tokens_jti", "refresh_tokens", ["jti"], unique=True)
    op.create_index("ix_refresh_tokens_family_id", "refresh_tokens", ["family_id"])
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])

    op.drop_table("token_blacklist")

    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column(
                "last_login_at_ms",
                sa.BigInteger(),
                nullable=True,
                comment="Latest successful sign-in [UTC epoch ms]",
            )
        )

    with op.batch_alter_table("audit_log") as batch:
        batch.add_column(
            sa.Column(
                "username",
                sa.String(64),
                nullable=True,
                comment="Name of the acting user",
            )
        )
        batch.add_column(
            sa.Column(
                "role",
                sa.String(32),
                nullable=True,
                comment="Role the user held when acting",
            )
        )
        batch.add_column(
            sa.Column(
                "outcome",
                sa.String(500),
                nullable=True,
                comment="How the action ended, e.g. accepted, refused: <reason>",
            )
        )
    op.create_index("ix_audit_log_username", "audit_log", ["username"])

    op.create_table(
        "scenario_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "kind",
            sa.String(16),
            nullable=False,
            comment="scenario | fault_injected | fault_cleared",
        ),
        sa.Column("scenario", sa.String(64), nullable=False),
        sa.Column(
            "run_id",
            sa.BigInteger(),
            nullable=False,
            comment="Physics run id after the action",
        ),
        sa.Column("fault_id", sa.String(64), nullable=True),
        sa.Column("fault_label", sa.String(128), nullable=True),
        sa.Column("severity", sa.Float(), nullable=True),
        sa.Column("simulation_time_s", sa.Float(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("at_ms", sa.BigInteger(), nullable=False, comment="[UTC epoch ms]"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "kind IN ('scenario', 'fault_injected', 'fault_cleared')",
            name="ck_scenario_runs_kind",
        ),
    )
    op.create_index("ix_scenario_runs_user_id", "scenario_runs", ["user_id"])
    op.create_index("ix_scenario_runs_at_ms", "scenario_runs", ["at_ms"])

    if _dialect() == "postgresql":
        _execute_all(POSTGRES_APPEND_ONLY)
    elif _dialect() == "sqlite":
        _execute_all(SQLITE_APPEND_ONLY)


def downgrade() -> None:
    if _dialect() == "postgresql":
        _execute_all(POSTGRES_DROP_APPEND_ONLY)
    elif _dialect() == "sqlite":
        _execute_all(SQLITE_DROP_APPEND_ONLY)

    op.drop_table("scenario_runs")

    op.drop_index("ix_audit_log_username", table_name="audit_log")
    with op.batch_alter_table("audit_log") as batch:
        batch.drop_column("outcome")
        batch.drop_column("role")
        batch.drop_column("username")

    with op.batch_alter_table("users") as batch:
        batch.drop_column("last_login_at_ms")

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

    op.drop_table("refresh_tokens")
