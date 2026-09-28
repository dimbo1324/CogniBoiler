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
from api_gateway.models.user import AuditLog

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

USER_ROW = (
    "INSERT INTO users (username, hashed_password, is_active, created_at_ms) "
    "VALUES (?, 'x', 1, 1)"
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


class TestCaseInsensitiveUsernames:
    def test_names_differing_only_in_case_are_refused(
        self, alembic_config: Config
    ) -> None:
        command.upgrade(alembic_config, "head")
        with connect(alembic_config) as db:
            db.execute(USER_ROW, ("Anna",))
            db.execute(USER_ROW, ("anna.b",))
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(USER_ROW, ("anna",))

    def test_the_downgrade_removes_only_that_rule(self, alembic_config: Config) -> None:
        command.upgrade(alembic_config, "head")
        command.downgrade(alembic_config, "0005")
        with connect(alembic_config) as db:
            db.execute(USER_ROW, ("Anna",))
            db.execute(USER_ROW, ("anna",))
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(USER_ROW, ("anna",))


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


class TestAuditAuthors:
    """Deleting a user must not rewrite the audit rows that name them (ARCH-15)."""

    def test_the_audit_log_keeps_its_authors(self) -> None:
        keep = revision_module("0007_audit_log_keeps_its_authors")
        assert keep.upgrade_statements() == [
            "ALTER TABLE audit_log DROP CONSTRAINT audit_log_user_id_fkey",
            "ALTER TABLE audit_log ADD CONSTRAINT audit_log_user_id_fkey "
            "FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE RESTRICT",
        ]

    def test_the_downgrade_restores_set_null(self) -> None:
        keep = revision_module("0007_audit_log_keeps_its_authors")
        assert keep.downgrade_statements()[-1].endswith("ON DELETE SET NULL")

    def test_the_model_declares_the_same_rule(self) -> None:
        (key,) = AuditLog.__table__.c.user_id.foreign_keys
        assert key.ondelete == "RESTRICT"
