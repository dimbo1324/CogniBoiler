"""The step that gives the application roles their passwords."""

from __future__ import annotations

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
