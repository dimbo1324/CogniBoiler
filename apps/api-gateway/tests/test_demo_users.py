"""Demo users are seeded only from configured passwords, never from code."""

from __future__ import annotations

import pytest
from api_gateway import db_init
from api_gateway.config import settings


def test_every_role_has_a_demo_user_named_after_it() -> None:
    roles = {name for name, _ in db_init.ROLE_DESCRIPTIONS}
    assert {username for username, _ in db_init.demo_users()} == roles


def test_demo_passwords_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_operator_password", "from-the-environment")
    passwords = dict(db_init.demo_users())
    assert passwords["operator"] == "from-the-environment"


def test_an_unset_password_is_empty_not_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field in (
        "demo_admin_password",
        "demo_engineer_password",
        "demo_operator_password",
        "demo_viewer_password",
    ):
        monkeypatch.setattr(settings, field, "")
    assert all(password == "" for _, password in db_init.demo_users())
