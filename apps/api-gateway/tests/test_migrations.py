"""The Alembic chain itself: it runs both ways, and it keeps the audit trail append-only.

The other suites build their tables from the ORM models, which carry no triggers and no
grants, so only this file exercises what the migrations enforce. SQLite runs the chain
for real; the PostgreSQL grants are checked as the statements the migrations issue.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from api_gateway.config import settings

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def revision_module(stem: str) -> ModuleType:
    path = MIGRATIONS / "versions" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"migration_{stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def alembic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    database = tmp_path / "chain.db"
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{database.as_posix()}"
    )
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.attributes["database"] = database
    return config


def connect(config: Config) -> sqlite3.Connection:
    database: Path = config.attributes["database"]
    return sqlite3.connect(database)


AUDIT_ROW = (
    "INSERT INTO audit_log (method, endpoint, response_status, duration_ms, "
    "timestamp_ms, ip_address) VALUES ('POST', '/auth/login', 401, 3, 1, '10.0.0.1')"
)


class TestTheChain:
    def test_it_has_one_head(self) -> None:
        config = Config()
        config.set_main_option("script_location", str(MIGRATIONS))
        assert len(ScriptDirectory.from_config(config).get_heads()) == 1

    def test_the_audit_log_takes_rows_and_refuses_to_change_them(
        self, alembic_config: Config
    ) -> None:
        command.upgrade(alembic_config, "head")
        with connect(alembic_config) as db:
            db.execute(AUDIT_ROW)
            assert db.execute("SELECT count(*) FROM audit_log").fetchone() == (1,)
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                db.execute("UPDATE audit_log SET response_status = 200")
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                db.execute("DELETE FROM audit_log")

    def test_it_downgrades_to_nothing_and_upgrades_again(
        self, alembic_config: Config
    ) -> None:
        command.upgrade(alembic_config, "head")
        command.downgrade(alembic_config, "base")
        with connect(alembic_config) as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        assert tables <= {"alembic_version"}
        command.upgrade(alembic_config, "head")
        with connect(alembic_config) as db:
            db.execute(AUDIT_ROW)
            with pytest.raises(sqlite3.DatabaseError, match="append-only"):
                db.execute("DELETE FROM audit_log")


class TestApplicationGrants:
    """PostgreSQL rights of the two application roles, as the chain leaves them."""

    def test_the_audit_log_is_only_read_and_appended_by_the_gateway(self) -> None:
        roles = revision_module("0004_application_roles")
        assert "audit_log" in roles.GATEWAY_APPEND_ONLY
        assert "audit_log" not in roles.GATEWAY_READ_WRITE
        assert "audit_log" not in roles.ALARMS_READ_WRITE

    def test_the_narrowing_takes_exactly_the_unused_rights(self) -> None:
        narrow = revision_module("0005_narrow_application_grants")
        assert narrow.revoke_statements() == [
            "REVOKE UPDATE, DELETE ON scenario_runs FROM cogniboiler_gateway",
            "REVOKE DELETE ON users, roles FROM cogniboiler_gateway",
            "REVOKE UPDATE, DELETE ON alarm_transitions FROM cogniboiler_alarms",
            "REVOKE DELETE ON alarm_events FROM cogniboiler_alarms",
        ]

    def test_the_downgrade_gives_back_what_the_upgrade_took(self) -> None:
        narrow = revision_module("0005_narrow_application_grants")
        taken = [
            statement.removeprefix("REVOKE ").replace(" FROM ", " | ")
            for statement in narrow.revoke_statements()
        ]
        given = [
            statement.removeprefix("GRANT ").replace(" TO ", " | ")
            for statement in narrow.grant_statements()
        ]
        assert taken == given

    def test_the_rights_the_code_uses_are_kept(self) -> None:
        narrow = revision_module("0005_narrow_application_grants")
        taken = " ".join(narrow.revoke_statements())
        # The gateway updates accounts (password, block, last sign-in) and replaces
        # role assignments; alert-manager updates the alarm lifecycle.
        assert "user_roles" not in taken
        assert "refresh_tokens" not in taken
        assert "UPDATE ON users" not in taken and "UPDATE, DELETE ON users" not in taken
        assert "UPDATE, DELETE ON alarm_events" not in taken
        assert "INSERT" not in taken and "SELECT" not in taken

    def test_no_migration_grants_what_would_undo_append_only(self) -> None:
        for path in sorted((MIGRATIONS / "versions").glob("0*.py")):
            text = path.read_text(encoding="utf-8").upper()
            for forbidden in ("GRANT TRUNCATE", "GRANT TRIGGER", "GRANT ALL"):
                assert forbidden not in text, (path.name, forbidden)
