"""Narrower grants: attribution trails can only be appended to

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-27

0004 gave both application roles SELECT, INSERT, UPDATE and DELETE on all their tables.
The code needs less, and a compromised service could otherwise rewrite who did what:

- cogniboiler_gateway only inserts into scenario_runs (who loaded a scenario, injected or
  cleared a fault) and never deletes an account or a role — accounts are blocked, never
  deleted. It loses UPDATE and DELETE on scenario_runs and DELETE on users and roles.
- cogniboiler_alarms only inserts into alarm_transitions (who acknowledged what, when)
  and never deletes an alarm. It loses UPDATE and DELETE on alarm_transitions and
  DELETE on alarm_events: a foreign-key cascade runs with the owner's rights, so deleting
  an alarm would still erase its transitions.

The downgrade gives exactly these rights back. PostgreSQL only; other databases (the test
suites' SQLite) have no roles.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GATEWAY_ROLE = "cogniboiler_gateway"
ALARMS_ROLE = "cogniboiler_alarms"

NARROWED: tuple[tuple[str, str, str], ...] = (
    ("UPDATE, DELETE", "scenario_runs", GATEWAY_ROLE),
    ("DELETE", "users, roles", GATEWAY_ROLE),
    ("UPDATE, DELETE", "alarm_transitions", ALARMS_ROLE),
    ("DELETE", "alarm_events", ALARMS_ROLE),
)


def revoke_statements() -> list[str]:
    return [
        f"REVOKE {rights} ON {tables} FROM {role}" for rights, tables, role in NARROWED
    ]


def grant_statements() -> list[str]:
    return [
        f"GRANT {rights} ON {tables} TO {role}" for rights, tables, role in NARROWED
    ]


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for statement in revoke_statements():
        op.execute(statement)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for statement in grant_statements():
        op.execute(statement)
