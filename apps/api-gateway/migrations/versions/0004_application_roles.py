"""Database roles for the applications: no DDL, and the audit log only appended to

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18

The gateway and alert-manager used to connect as the owner of every table, and an owner
can disable the triggers that keep audit_log append-only. From here on only the migration
job connects as the owner. Two roles get exactly what their service does:

- cogniboiler_gateway: read and write the account, session and scenario tables; read and
  insert audit_log, nothing else on it;
- cogniboiler_alarms: read and write the alarm tables.

Neither owns a table, so neither can alter one, disable a trigger or truncate. The roles
are created without login here: a migration must not hold a password.
`python -m api_gateway.db_roles`, run by the same job, gives them a login and the
passwords from the environment. A later table grants its rights in its own migration.

PostgreSQL only; other databases (the test suites' SQLite) have no roles.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GATEWAY_ROLE = "cogniboiler_gateway"
ALARMS_ROLE = "cogniboiler_alarms"

GATEWAY_READ_WRITE = ("users", "roles", "user_roles", "refresh_tokens", "scenario_runs")
GATEWAY_APPEND_ONLY = ("audit_log",)
ALARMS_READ_WRITE = ("alarm_events", "alarm_transitions")


def _create_role(role: str) -> str:
    return f"""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
            CREATE ROLE {role} NOLOGIN;
        END IF;
    END
    $$
    """


def _grant_sequences(role: str, tables: Sequence[str]) -> list[str]:
    return [
        f"GRANT USAGE, SELECT ON SEQUENCE {sequence} TO {role}"
        for sequence in (f"{table}_id_seq" for table in tables)
    ]


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    statements = [
        _create_role(GATEWAY_ROLE),
        _create_role(ALARMS_ROLE),
        # Only the owner creates objects; PostgreSQL 15+ already denies it to PUBLIC.
        "REVOKE CREATE ON SCHEMA public FROM PUBLIC",
        f"GRANT USAGE ON SCHEMA public TO {GATEWAY_ROLE}, {ALARMS_ROLE}",
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(GATEWAY_READ_WRITE)} "
        f"TO {GATEWAY_ROLE}",
        f"GRANT SELECT, INSERT ON {', '.join(GATEWAY_APPEND_ONLY)} TO {GATEWAY_ROLE}",
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(ALARMS_READ_WRITE)} "
        f"TO {ALARMS_ROLE}",
        *_grant_sequences(GATEWAY_ROLE, GATEWAY_READ_WRITE + GATEWAY_APPEND_ONLY),
        *_grant_sequences(ALARMS_ROLE, ALARMS_READ_WRITE),
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for role in (GATEWAY_ROLE, ALARMS_ROLE):
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role};
                    REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM {role};
                    REVOKE ALL ON SCHEMA public FROM {role};
                    DROP ROLE {role};
                END IF;
            END
            $$
            """
        )
