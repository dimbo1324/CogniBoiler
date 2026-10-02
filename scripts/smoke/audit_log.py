"""Prove on the live PostgreSQL that the audit log cannot be rewritten (invariant I10).

The unit suites build their schema from the ORM models, which carry neither the grants
of migration 0004 nor the triggers of 0003, so only a real database can show them. The
check runs psql inside the PostgreSQL container as the owner, which already holds its
credentials, so no password reaches a command line. Everything happens in one
transaction that is rolled back, and ``ON_ERROR_ROLLBACK`` wraps each statement in a
savepoint, so one refusal does not hide the next and nothing is ever changed:

* as the gateway role — what the gateway connects as — reading is allowed, and
  ``UPDATE``, ``DELETE`` and ``TRUNCATE`` are refused by the grants ("permission
  denied"); the ``WHERE false`` forms touch no row, so a refusal can only come from the
  grant, not from the row trigger;
* as the owner, who holds every privilege, ``UPDATE`` of the newest row and ``TRUNCATE``
  are refused by the append-only triggers.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts._toolkit.compose import exec_sh
from scripts._toolkit.processes import NOT_FOUND, CommandResult, run_piped

MARKER = "@check "
END = "end"
GRANT_REFUSAL = "permission denied for table"
TRIGGER_REFUSAL = "is append-only"

#: (check, statement, the text its refusal must carry; None means it must succeed)
GATEWAY_STATEMENTS: tuple[tuple[str, str, str | None], ...] = (
    ("reads", "SELECT count(*) FROM {table};", None),
    ("UPDATE", "UPDATE {table} SET id = id WHERE false;", GRANT_REFUSAL),
    ("DELETE", "DELETE FROM {table} WHERE false;", GRANT_REFUSAL),
    ("TRUNCATE", "TRUNCATE {table};", GRANT_REFUSAL),
)
OWNER_STATEMENTS: tuple[tuple[str, str, str | None], ...] = (
    (
        "UPDATE",
        "UPDATE {table} SET id = id WHERE id = (SELECT max(id) FROM {table});",
        TRIGGER_REFUSAL,
    ),
    ("TRUNCATE", "TRUNCATE {table};", TRIGGER_REFUSAL),
)


@dataclass(frozen=True)
class Outcome:
    name: str
    passed: bool
    detail: str


def psql_command(config: dict[str, Any]) -> list[str]:
    """psql in the PostgreSQL container as its owner, the script on stdin, errors on
    stdout so the transcript keeps them next to the check they belong to."""
    return exec_sh(
        config,
        str(config["postgres_service"]),
        'psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" '
        "--no-psqlrc --quiet 2>&1",
    )


def _checks(role: str) -> list[tuple[str, str, str | None]]:
    return [
        *(
            (f"{role} {name}", sql, refusal)
            for name, sql, refusal in GATEWAY_STATEMENTS
        ),
        *((f"owner {name}", sql, refusal) for name, sql, refusal in OWNER_STATEMENTS),
    ]


def psql_script(role: str, table: str) -> str:
    lines = [
        "\\set ON_ERROR_STOP off",
        "\\set ON_ERROR_ROLLBACK on",
        "BEGIN;",
        f"SET LOCAL ROLE {role};",
    ]
    for index, (name, sql, _) in enumerate(_checks(role)):
        if index == len(GATEWAY_STATEMENTS):
            lines.append("RESET ROLE;")
        lines.append(f"\\echo {MARKER}{name}")
        lines.append(sql.format(table=table))
    lines.extend([f"\\echo {MARKER}{END}", "ROLLBACK;", ""])
    return "\n".join(lines)


def judge(role: str, transcript: str) -> list[Outcome]:
    """Read the psql transcript back into one outcome per check."""
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in transcript.splitlines():
        if line.startswith(MARKER):
            current = line[len(MARKER) :].strip()
            sections[current] = []
        else:
            sections[current].append(line)
    preamble_errors = [line for line in sections[""] if "ERROR" in line]
    checks = _checks(role)
    if preamble_errors or END not in sections:
        reason = (
            preamble_errors[0].strip()
            if preamble_errors
            else "psql did not run the whole check"
        )
        return [Outcome(name, False, reason) for name, _, _ in checks]

    outcomes = []
    for name, _, refusal in checks:
        errors = [line.strip() for line in sections.get(name, []) if "ERROR" in line]
        if refusal is None:
            passed = not errors
            detail = "allowed" if passed else errors[0]
        else:
            passed = any(refusal in error for error in errors)
            detail = (
                f"refused ({refusal})"
                if passed
                else (errors[0] if errors else "NOT refused")
            )
        outcomes.append(Outcome(name, passed, detail))
    return outcomes


Runner = Callable[..., CommandResult]


def check_append_only(
    config: dict[str, Any], root: Path, run: Runner = run_piped
) -> list[Outcome]:
    role = str(config["gateway_role"])
    table = str(config["audit_table"])
    with tempfile.TemporaryDirectory() as temp:
        script = Path(temp) / "check.sql"
        transcript = Path(temp) / "transcript.txt"
        script.write_text(psql_script(role, table), encoding="utf-8")
        result = run(psql_command(config), root, stdin=script, stdout=transcript)
        text = (
            transcript.read_text(encoding="utf-8", errors="replace")
            if transcript.exists()
            else ""
        )
    if result.returncode == NOT_FOUND:
        reason = "docker is not on PATH"
        return [Outcome(name, False, reason) for name, _, _ in _checks(role)]
    if not result.ok and f"{MARKER}{END}" not in text:
        reason = f"psql in {config['postgres_service']} exited {result.returncode}"
        return [Outcome(name, False, reason) for name, _, _ in _checks(role)]
    return judge(role, text)
