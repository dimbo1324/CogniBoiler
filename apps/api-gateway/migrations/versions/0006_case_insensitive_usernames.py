"""Usernames are unique ignoring case

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-27

Account creation refuses "Anna" when "anna" exists, but only by a check before the
insert; two creations at the same moment both passed it, because ix_users_username
compares case-sensitively. A unique index on lower(username) makes the database refuse
the second one, which the gateway already answers with 409 users.username_taken.

The upgrade fails if the table already holds two names that differ only in case; rename
or block one of them first.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "ix_users_username_lower"


def upgrade() -> None:
    op.create_index(INDEX, "users", [sa.text("lower(username)")], unique=True)


def downgrade() -> None:
    op.drop_index(INDEX, table_name="users")
