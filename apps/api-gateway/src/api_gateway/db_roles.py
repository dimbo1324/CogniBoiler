"""
Give the application roles a login and their passwords.

Migration 0004 creates `cogniboiler_gateway` and `cogniboiler_alarms` without a login,
because a migration must not hold a secret. The migration job runs this right after
`alembic upgrade head`, connected as the owner, with the passwords in its environment:

    GATEWAY_DB_PASSWORD, ALARMS_DB_PASSWORD

Run it again after changing a password in .env. The statement is built by PostgreSQL's own
`format(%I, %L)`, so a password is always quoted as a literal and never logged here.

The statement that is finally executed holds the password as a literal, and SQLAlchemy
puts the text of a failed statement into its error message. A failure is therefore
reported by the error's type only, and the error itself is dropped, not chained.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from cogniboiler_observability import configure_logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from api_gateway.config import settings

logger = logging.getLogger(__name__)

ROLE_PASSWORDS = {
    "cogniboiler_gateway": "GATEWAY_DB_PASSWORD",
    "cogniboiler_alarms": "ALARMS_DB_PASSWORD",
}


async def _give_login(connection: AsyncConnection, role: str, password: str) -> None:
    failure: str | None = None
    try:
        statement = await connection.scalar(
            text(
                "SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', "
                "CAST(:role AS text), CAST(:pw AS text))"
            ),
            {"role": role, "pw": password},
        )
        # Sent as is: text() would read a colon in a password as a parameter.
        await connection.exec_driver_sql(str(statement))
    except SQLAlchemyError as exc:
        failure = type(exc).__name__
    if failure is not None:
        # Raised outside the except block, so the error with the statement text is
        # neither the cause nor the context of this one.
        logger.error("Could not give role %s a login: %s", role, failure)
        raise RuntimeError(f"giving {role} a login failed ({failure})")
    logger.info("Role %s can sign in", role)


async def provision(database_url: str, passwords: dict[str, str]) -> None:
    # Errors must not echo the statement's parameters: one of them is a password.
    engine = create_async_engine(database_url, hide_parameters=True)
    try:
        async with engine.begin() as connection:
            for role, password in passwords.items():
                await _give_login(connection, role, password)
    finally:
        await engine.dispose()


def main() -> int:
    configure_logging("db-roles")
    passwords: dict[str, str] = {}
    for role, variable in ROLE_PASSWORDS.items():
        value = os.environ.get(variable, "")
        if not value:
            logger.error("%s is empty; run dev-secrets", variable)
            return 1
        passwords[role] = value
    try:
        asyncio.run(provision(settings.database_url, passwords))
    except RuntimeError as exc:
        logger.error("Provisioning the application roles failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
