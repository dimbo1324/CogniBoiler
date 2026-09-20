"""The step that gives the application roles their passwords.

A generated password is arbitrary text: it can hold a quote, a backslash, a colon or a
percent sign. Every one of those has meaning somewhere on the way to PostgreSQL — in SQL,
in SQLAlchemy's `text()` parameters, in `%`-formatting — so this step builds the statement
on the server with `format('%I', '%L')` and never interpolates anything itself.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import pytest
from api_gateway import db_roles


def test_it_refuses_to_run_without_every_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_DB_PASSWORD", "a-generated-password")
    monkeypatch.delenv("ALARMS_DB_PASSWORD", raising=False)
    called: list[object] = []

    async def provision(*args: object) -> None:
        called.append(args)

    monkeypatch.setattr(db_roles, "provision", provision)
    assert db_roles.main() == 1
    assert called == []


def test_both_roles_get_their_own_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_DB_PASSWORD", "gateway-secret")
    monkeypatch.setenv("ALARMS_DB_PASSWORD", "alarms-secret")
    seen: dict[str, str] = {}

    async def provision(_: str, passwords: dict[str, str]) -> None:
        seen.update(passwords)

    monkeypatch.setattr(db_roles, "provision", provision)
    assert db_roles.main() == 0
    assert seen == {
        "cogniboiler_gateway": "gateway-secret",
        "cogniboiler_alarms": "alarms-secret",
    }


class FakeConnection:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    async def scalar(self, statement: object, parameters: dict[str, str]) -> str:
        self._engine.asked.append((str(statement), dict(parameters)))
        # What PostgreSQL's format('%I', '%L') would give back, quoting included.
        role, password = parameters["role"], parameters["pw"]
        quoted = password.replace("'", "''")
        return f"ALTER ROLE \"{role}\" WITH LOGIN PASSWORD '{quoted}'"

    async def exec_driver_sql(self, statement: str) -> None:
        if self._engine.fail_on_exec:
            raise RuntimeError("connection lost")
        self._engine.executed.append(statement)


class FakeBegin:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    async def __aenter__(self) -> FakeConnection:
        return FakeConnection(self._engine)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class FakeEngine:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.asked: list[tuple[str, dict[str, str]]] = []
        self.executed: list[str] = []
        self.disposed = 0
        self.fail_on_exec = False

    def begin(self) -> FakeBegin:
        return FakeBegin(self)

    async def dispose(self) -> None:
        self.disposed += 1


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    made: list[FakeEngine] = []

    def create(url: str, **options: Any) -> FakeEngine:
        made.append(FakeEngine(url=url, **options))
        return made[-1]

    monkeypatch.setattr(db_roles, "create_async_engine", create)
    return made


HOSTILE = "p'a\"s;s--\\w%o:rd"


class TestProvisioning:
    async def test_the_statement_is_built_by_postgresql_not_by_us(
        self, engine: list[FakeEngine]
    ) -> None:
        await db_roles.provision(
            "postgresql+asyncpg://owner@db/cogniboiler",
            {"cogniboiler_gateway": HOSTILE},
        )
        (used,) = engine
        (asked, parameters) = used.asked[0]
        # The password travels as a bound parameter, never inside the SQL we send.
        assert "format(" in asked
        assert "%I" in asked and "%L" in asked
        assert HOSTILE not in asked
        assert parameters == {"role": "cogniboiler_gateway", "pw": HOSTILE}

    async def test_the_answer_is_sent_as_it_came_back(
        self, engine: list[FakeEngine]
    ) -> None:
        # exec_driver_sql, not text(): a colon in a password would otherwise be read as
        # the name of a bind parameter and the statement would fail — or bind something.
        await db_roles.provision("postgresql+asyncpg://owner@db/x", {"role_a": HOSTILE})
        (used,) = engine
        quoted = "'" + HOSTILE.replace("'", "''") + "'"
        assert used.executed == ['ALTER ROLE "role_a" WITH LOGIN PASSWORD ' + quoted]

    async def test_a_failure_never_echoes_the_parameters(
        self, engine: list[FakeEngine]
    ) -> None:
        await db_roles.provision("postgresql+asyncpg://owner@db/x", {"role_a": "pw"})
        (used,) = engine
        assert used.options["hide_parameters"] is True

    async def test_every_role_is_provisioned_in_turn(
        self, engine: list[FakeEngine]
    ) -> None:
        await db_roles.provision(
            "postgresql+asyncpg://owner@db/x",
            {"cogniboiler_gateway": "one", "cogniboiler_alarms": "two"},
        )
        (used,) = engine
        assert [
            role for _, parameters in used.asked for role in [parameters["role"]]
        ] == [
            "cogniboiler_gateway",
            "cogniboiler_alarms",
        ]
        assert len(used.executed) == 2

    async def test_the_engine_is_disposed_even_when_the_statement_fails(
        self, engine: list[FakeEngine], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def create(url: str, **options: Any) -> FakeEngine:
            made = FakeEngine(url=url, **options)
            made.fail_on_exec = True
            engine.append(made)
            return made

        monkeypatch.setattr(db_roles, "create_async_engine", create)
        with pytest.raises(RuntimeError, match="connection lost"):
            await db_roles.provision("postgresql+asyncpg://owner@db/x", {"r": "pw"})
        (used,) = engine
        assert used.disposed == 1

    async def test_no_roles_asks_nothing_and_still_closes(
        self, engine: list[FakeEngine]
    ) -> None:
        await db_roles.provision("postgresql+asyncpg://owner@db/x", {})
        (used,) = engine
        assert (used.asked, used.executed, used.disposed) == ([], [], 1)
