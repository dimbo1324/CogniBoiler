"""Alarm lifecycle: alarm_events and alarm_transitions owned by alert-manager

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-15

Before this revision alert-manager created a flat alarm_events table itself with
create_all, outside the migration chain: one row per PLC message, no lifecycle. If that
table exists it is dropped — it holds only development data, its rows cannot be mapped to
alarm lifecycles, and its index, key and sequence names would collide with the new table.
The downgrade removes the lifecycle tables and does not restore the flat one.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OPEN_ALARM_PREDICATE = "state <> 'CLEARED'"


def _drop_flat_alarm_table() -> None:
    if context.is_offline_mode():
        return
    inspector = sa.inspect(op.get_bind())
    if "alarm_events" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("alarm_events")}
    if "state" not in columns:
        op.drop_table("alarm_events")


def upgrade() -> None:
    _drop_flat_alarm_table()

    op.create_table(
        "alarm_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "key",
            sa.String(200),
            nullable=False,
            comment="Condition identity: source:parameter:direction:severity",
        ),
        sa.Column("source_service", sa.String(64), nullable=False),
        sa.Column("parameter", sa.String(128), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("unit", sa.String(16), nullable=False),
        sa.Column(
            "state",
            sa.String(16),
            nullable=False,
            comment="ACTIVE_UNACK | ACTIVE_ACK | CLEARED_UNACK | CLEARED",
        ),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("topic", sa.String(128), nullable=False),
        sa.Column("value", sa.Float(), nullable=False, comment="Latest value, SI"),
        sa.Column("threshold", sa.Float(), nullable=False, comment="Limit, SI"),
        sa.Column(
            "raised_at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="Condition became active [UTC epoch ms]",
        ),
        sa.Column(
            "cleared_at_ms",
            sa.BigInteger(),
            nullable=True,
            comment="Condition ended [UTC epoch ms]",
        ),
        sa.Column(
            "acknowledged_at_ms",
            sa.BigInteger(),
            nullable=True,
            comment="Acknowledged [UTC epoch ms]",
        ),
        sa.Column("acknowledged_by", sa.String(128), nullable=True),
        sa.Column("ack_comment", sa.String(500), nullable=True),
        sa.Column(
            "occurrence_count",
            sa.Integer(),
            nullable=False,
            comment="Times the condition became active while this alarm was open",
        ),
        sa.Column("updated_at_ms", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "state IN ('ACTIVE_UNACK', 'ACTIVE_ACK', 'CLEARED_UNACK', 'CLEARED')",
            name="ck_alarm_events_state",
        ),
        sa.CheckConstraint(
            "severity IN ('warning', 'critical')", name="ck_alarm_events_severity"
        ),
    )
    op.create_index("ix_alarm_events_key", "alarm_events", ["key"])
    op.create_index("ix_alarm_events_parameter", "alarm_events", ["parameter"])
    op.create_index("ix_alarm_events_severity", "alarm_events", ["severity"])
    op.create_index("ix_alarm_events_raised_at_ms", "alarm_events", ["raised_at_ms"])
    op.create_index(
        "ix_alarm_events_state_raised", "alarm_events", ["state", "raised_at_ms"]
    )
    # At most one open alarm per condition, enforced by the database.
    op.create_index(
        "uq_alarm_events_open_key",
        "alarm_events",
        ["key"],
        unique=True,
        postgresql_where=sa.text(OPEN_ALARM_PREDICATE),
        sqlite_where=sa.text(OPEN_ALARM_PREDICATE),
    )

    op.create_table(
        "alarm_transitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("alarm_id", sa.Integer(), nullable=False),
        sa.Column("from_state", sa.String(16), nullable=True),
        sa.Column("to_state", sa.String(16), nullable=False),
        sa.Column(
            "at_ms",
            sa.BigInteger(),
            nullable=False,
            comment="When the state changed [UTC epoch ms]",
        ),
        sa.Column(
            "actor",
            sa.String(128),
            nullable=False,
            comment="Source service, or the user who acknowledged",
        ),
        sa.Column("comment", sa.String(500), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["alarm_id"], ["alarm_events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alarm_transitions_alarm_id", "alarm_transitions", ["alarm_id"])
    op.create_index("ix_alarm_transitions_at_ms", "alarm_transitions", ["at_ms"])


def downgrade() -> None:
    op.drop_table("alarm_transitions")
    op.drop_table("alarm_events")
