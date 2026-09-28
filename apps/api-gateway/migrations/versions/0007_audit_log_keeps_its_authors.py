"""The audit log keeps its authors: deleting a user with audit rows is refused

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-28

0001 declared audit_log.user_id ON DELETE SET NULL, which asks the database to rewrite
audit rows when a user is deleted. PostgreSQL runs SET NULL as an UPDATE of the
referencing rows, which the append-only trigger of 0003 refuses, so such a delete already
failed, with a confusing "append-only" error. RESTRICT says the intent: an account with
an audit history is blocked, never deleted. No route deletes users.

PostgreSQL only, like 0004 and 0005: the test suites' SQLite does not enforce foreign
keys, and changing one there would rebuild the table without its triggers.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "audit_log_user_id_fkey"


def _statements(on_delete: str) -> list[str]:
    return [
        f"ALTER TABLE audit_log DROP CONSTRAINT {CONSTRAINT}",
        f"ALTER TABLE audit_log ADD CONSTRAINT {CONSTRAINT} "
        f"FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE {on_delete}",
    ]


def upgrade_statements() -> list[str]:
    return _statements("RESTRICT")


def downgrade_statements() -> list[str]:
    return _statements("SET NULL")


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for statement in upgrade_statements():
        op.execute(statement)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for statement in downgrade_statements():
        op.execute(statement)
