"""The smoke step that proves on PostgreSQL that the audit log cannot be rewritten.

The transcripts below are psql's real output, recorded against a throwaway PostgreSQL 16
holding the triggers of migration 0003 and the grants of 0004 — once intact, once with
UPDATE granted to the gateway and the TRUNCATE trigger dropped, once for a role that does
not exist.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts._toolkit.processes import NOT_FOUND, CommandResult
from scripts.smoke import __main__ as smoke
from scripts.smoke.audit_log import (
    Outcome,
    check_append_only,
    judge,
    psql_command,
    psql_script,
)

ROLE = "cogniboiler_gateway"
CONFIG = {
    "project_name": "cogniboiler",
    "compose_file": "docker-compose.yml",
    "postgres_service": "postgresql",
    "gateway_role": ROLE,
    "audit_table": "audit_log",
}

INTACT = """\
@check cogniboiler_gateway reads
 count
-------
     2
(1 row)

@check cogniboiler_gateway UPDATE
ERROR:  permission denied for table audit_log
@check cogniboiler_gateway DELETE
ERROR:  permission denied for table audit_log
@check cogniboiler_gateway TRUNCATE
ERROR:  permission denied for table audit_log
@check owner UPDATE
ERROR:  audit_log is append-only: UPDATE refused
CONTEXT:  PL/pgSQL function audit_log_append_only() line 3 at RAISE
@check owner TRUNCATE
ERROR:  audit_log is append-only: TRUNCATE refused
CONTEXT:  PL/pgSQL function audit_log_append_only() line 3 at RAISE
@check end
"""

BROKEN = """\
@check cogniboiler_gateway reads
 count
-------
     2
(1 row)

@check cogniboiler_gateway UPDATE
@check cogniboiler_gateway DELETE
ERROR:  permission denied for table audit_log
@check cogniboiler_gateway TRUNCATE
ERROR:  permission denied for table audit_log
@check owner UPDATE
ERROR:  audit_log is append-only: UPDATE refused
CONTEXT:  PL/pgSQL function audit_log_append_only() line 3 at RAISE
@check owner TRUNCATE
@check end
"""

NO_ROLE = """\
ERROR:  role "cogniboiler_gateway" does not exist
@check cogniboiler_gateway reads
 count
-------
     2
(1 row)

@check end
"""


def _verdicts(outcomes: list[Outcome]) -> dict[str, bool]:
    return {outcome.name: outcome.passed for outcome in outcomes}


class ScriptTest(unittest.TestCase):
    def test_everything_runs_in_one_transaction_that_is_rolled_back(self) -> None:
        lines = psql_script(ROLE, "audit_log").splitlines()
        self.assertIn("\\set ON_ERROR_ROLLBACK on", lines)
        self.assertLess(lines.index("BEGIN;"), lines.index(f"SET LOCAL ROLE {ROLE};"))
        self.assertEqual(lines[-1], "ROLLBACK;")
        self.assertNotIn("COMMIT;", lines)

    def test_the_gateway_statements_touch_no_row(self) -> None:
        # A refusal of these can only come from the grant, never from the row trigger.
        script = psql_script(ROLE, "audit_log")
        gateway_part = script.split("RESET ROLE;")[0]
        self.assertIn("UPDATE audit_log SET id = id WHERE false;", gateway_part)
        self.assertIn("DELETE FROM audit_log WHERE false;", gateway_part)

    def test_psql_runs_inside_the_container_without_a_password(self) -> None:
        command = psql_command(CONFIG)
        self.assertEqual(command[6:10], ["exec", "-T", "postgresql", "sh"])
        self.assertIn('"$POSTGRES_USER"', command[-1])
        self.assertNotIn("PASSWORD", " ".join(command))


class JudgeTest(unittest.TestCase):
    def test_an_intact_database_passes_every_check(self) -> None:
        verdicts = _verdicts(judge(ROLE, INTACT))
        self.assertEqual(len(verdicts), 6)
        self.assertTrue(all(verdicts.values()), verdicts)

    def test_a_loosened_grant_and_a_dropped_trigger_are_named(self) -> None:
        verdicts = _verdicts(judge(ROLE, BROKEN))
        self.assertFalse(verdicts[f"{ROLE} UPDATE"])
        self.assertFalse(verdicts["owner TRUNCATE"])
        self.assertTrue(verdicts[f"{ROLE} DELETE"])
        self.assertTrue(verdicts["owner UPDATE"])

    def test_a_missing_role_fails_every_check_with_its_reason(self) -> None:
        outcomes = judge(ROLE, NO_ROLE)
        self.assertFalse(any(outcome.passed for outcome in outcomes))
        self.assertIn("does not exist", outcomes[0].detail)

    def test_a_transcript_cut_short_fails_every_check(self) -> None:
        cut = INTACT.split("@check owner UPDATE")[0]
        self.assertFalse(any(outcome.passed for outcome in judge(ROLE, cut)))

    def test_a_trigger_refusal_does_not_count_as_a_grant_refusal(self) -> None:
        # The trigger raises the same SQLSTATE (insufficient_privilege); only the
        # grant's message proves the gateway lacks the privilege itself.
        trigger_only = INTACT.replace(
            "@check cogniboiler_gateway TRUNCATE\n"
            "ERROR:  permission denied for table audit_log",
            "@check cogniboiler_gateway TRUNCATE\n"
            "ERROR:  audit_log is append-only: TRUNCATE refused",
        )
        self.assertFalse(_verdicts(judge(ROLE, trigger_only))[f"{ROLE} TRUNCATE"])


class FakePsql:
    def __init__(self, transcript: str, returncode: int = 0) -> None:
        self.transcript = transcript
        self.returncode = returncode
        self.script = ""

    def __call__(
        self, argv: list[str], cwd: Path, *, stdin: Path, stdout: Path
    ) -> CommandResult:
        self.script = stdin.read_text(encoding="utf-8")
        if self.returncode != NOT_FOUND:
            stdout.write_text(self.transcript, encoding="utf-8")
        return CommandResult(argv, self.returncode)


class CheckTest(unittest.TestCase):
    def test_the_script_is_fed_to_psql_and_its_transcript_judged(self) -> None:
        fake = FakePsql(INTACT)
        outcomes = check_append_only(CONFIG, Path("."), run=fake)
        self.assertIn(f"SET LOCAL ROLE {ROLE};", fake.script)
        self.assertTrue(all(outcome.passed for outcome in outcomes))

    def test_without_docker_every_check_fails(self) -> None:
        outcomes = check_append_only(CONFIG, Path("."), run=FakePsql("", NOT_FOUND))
        self.assertFalse(any(outcome.passed for outcome in outcomes))
        self.assertIn("docker", outcomes[0].detail)

    def test_a_failing_psql_without_a_transcript_fails_every_check(self) -> None:
        outcomes = check_append_only(CONFIG, Path("."), run=FakePsql("", 1))
        self.assertFalse(any(outcome.passed for outcome in outcomes))
        self.assertIn("exited 1", outcomes[0].detail)


class SmokeMainTest(unittest.TestCase):
    def _main(self, *argv: str, outcomes: list[Outcome]) -> tuple[int, mock.Mock]:
        with (
            mock.patch.object(smoke, "load_env", return_value={}),
            mock.patch.object(smoke, "run_checks", return_value=smoke.Report()),
            mock.patch.object(
                smoke, "check_append_only", return_value=outcomes
            ) as checked,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return smoke.main(list(argv)), checked

    def test_a_refusal_missing_on_postgresql_fails_the_smoke(self) -> None:
        bad: list[Any] = [Outcome(f"{ROLE} UPDATE", False, "NOT refused")]
        code, checked = self._main(outcomes=bad)
        self.assertEqual(code, 1)
        checked.assert_called_once()

    def test_an_intact_audit_log_passes(self) -> None:
        good = [Outcome(f"{ROLE} UPDATE", True, "refused")]
        code, _ = self._main(outcomes=good)
        self.assertEqual(code, 0)

    def test_the_database_check_can_be_skipped_for_a_remote_gateway(self) -> None:
        code, checked = self._main("--skip-database", outcomes=[])
        self.assertEqual(code, 0)
        checked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
