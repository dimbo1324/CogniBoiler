"""The audit trail: what is recorded, how it is read back, and what it never holds."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from api_gateway.audit import (
    carries_credentials,
    command_outcome,
    should_audit,
    write_audit_entry,
)
from api_gateway.auth.jwt_handler import issue_refresh_token
from api_gateway.config import settings
from api_gateway.dependencies import get_db
from api_gateway.models.user import AuditLog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

NEW_PASSWORD = "Copper-Kettle-Bridge-7"


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


async def entries(
    client: AsyncClient, admin: dict[str, str], **params: Any
) -> list[dict[str, Any]]:
    response = await client.get("/api/v1/audit", params=params, headers=bearer(admin))
    assert response.status_code == 200, response.text
    items: list[dict[str, Any]] = response.json()["items"]
    return items


class TestWhatIsRecorded:
    @pytest.mark.parametrize(
        ("method", "path", "status", "expected"),
        [
            ("POST", "/api/v1/commands/load", 200, True),
            ("DELETE", "/api/v1/simulation/faults", 200, True),
            ("GET", "/api/v1/status", 200, False),
            ("GET", "/api/v1/status", 401, True),
            ("GET", "/api/v1/plant", 403, True),
            ("GET", "/api/v1/history", 429, True),
            ("GET", "/api/v1/users", 200, True),
            ("GET", "/api/v1/audit", 200, True),
            ("GET", "/health", 200, False),
            ("POST", "/health", 405, False),
            ("GET", "/ready", 503, False),
        ],
    )
    def test_the_policy(
        self, method: str, path: str, status: int, expected: bool
    ) -> None:
        assert should_audit(method, path, status) is expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/auth/login", True),
            ("/auth/password", True),
            ("/api/v1/users", True),
            ("/api/v1/users/7/password", True),
            ("/auth/login/", True),
            ("/Auth/Login", True),
            ("/auth/refresh", False),
            ("/auth/logout", False),
            ("/api/v1/users/7", False),
            ("/api/v1/users/7/revoke-sessions", False),
            ("/api/v1/commands/load", False),
        ],
    )
    def test_which_bodies_carry_credentials(self, path: str, expected: bool) -> None:
        assert carries_credentials(path) is expected

    def test_a_command_outcome_names_the_refusal(self) -> None:
        assert command_outcome(True, "") == "accepted"
        assert command_outcome(False, "E-Stop active") == "refused: E-Stop active"

    async def test_a_sign_in_is_recorded_with_its_actor_and_outcome(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        found = await entries(client, admin_tokens, endpoint="/auth/login")
        assert found[0]["username"] == "viewer1"
        assert found[0]["role"] == "viewer"
        assert found[0]["outcome"] == "signed in"
        assert found[0]["detail"] == "username=viewer1"
        assert found[0]["response_status"] == 200

    async def test_a_command_body_is_kept_only_as_a_digest(
        self,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        body = json.dumps({"mode": "manual"}).encode()
        await client.post(
            "/api/v1/commands/mode",
            content=body,
            headers={"Content-Type": "application/json", **bearer(operator_tokens)},
        )
        row = (await entries(client, admin_tokens, endpoint="/api/v1/commands"))[0]
        assert row["request_body_hash"] == hashlib.sha256(body).hexdigest()
        assert "manual" not in json.dumps(row)

    async def test_a_sign_in_body_leaves_no_digest(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        # A digest of {"username": ..., "password": ...} is an unsalted, fast password
        # verifier: whoever reads the audit log could crack it offline.
        body = json.dumps(
            {"username": "viewer1", "password": "wrong_password"}
        ).encode()
        await client.post(
            "/auth/login", content=body, headers={"Content-Type": "application/json"}
        )
        row = (await entries(client, admin_tokens, endpoint="/auth/login"))[0]
        assert row["request_body_hash"] is None
        assert row["outcome"] == "refused: invalid credentials"
        assert "wrong_password" not in json.dumps(row)

    async def test_a_misspelt_sign_in_path_leaves_no_digest_either(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/auth/login/", json={"username": "viewer1", "password": "wrong_password"}
        )
        await client.post(
            "/AUTH/LOGIN", json={"username": "viewer1", "password": "wrong_password"}
        )
        rows = await entries(client, admin_tokens, method="POST")
        assert {row["endpoint"] for row in rows} >= {"/auth/login/", "/AUTH/LOGIN"}
        assert all(row["request_body_hash"] is None for row in rows)

    async def test_a_password_change_leaves_no_digest(
        self,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        await client.post(
            "/auth/password",
            json={"current_password": "not-my-password", "new_password": NEW_PASSWORD},
            headers=bearer(operator_tokens),
        )
        row = (await entries(client, admin_tokens, endpoint="/auth/password"))[0]
        assert row["request_body_hash"] is None

    async def test_an_account_creation_and_a_reset_leave_no_digest(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        created = await client.post(
            "/api/v1/users",
            json={"username": "newbie", "password": NEW_PASSWORD, "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert created.status_code == 201, created.text
        reset = await client.post(
            f"/api/v1/users/{created.json()['id']}/password",
            json={"new_password": "Another-Harbour-Lamp-9"},
            headers=bearer(admin_tokens),
        )
        assert reset.status_code == 200, reset.text
        rows = await entries(client, admin_tokens, method="POST")
        endpoints = {row["endpoint"] for row in rows}
        assert "/api/v1/users" in endpoints
        assert f"/api/v1/users/{created.json()['id']}/password" in endpoints
        assert all(row["request_body_hash"] is None for row in rows)

    async def test_a_refused_read_is_recorded_with_the_caller(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        refused = await client.get("/api/v1/audit", headers=bearer(viewer_tokens))
        assert refused.status_code == 403
        row = (await entries(client, admin_tokens, status=403))[0]
        assert row["username"] == "viewer1"
        assert row["method"] == "GET"

    async def test_an_anonymous_refusal_is_recorded_without_a_user(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.get("/api/v1/status")
        row = (await entries(client, admin_tokens, status=401))[0]
        assert row["user_id"] is None
        assert row["endpoint"] == "/api/v1/status"

    async def test_telemetry_reads_are_not_recorded(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        await client.get("/api/v1/status", headers=bearer(viewer_tokens))
        assert await entries(client, admin_tokens, endpoint="/api/v1/status") == []

    async def test_a_plc_refusal_is_recorded_as_such(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        app.state.plc_client.accept = False
        app.state.plc_client.reason = "E-Stop active"
        response = await client.post(
            "/api/v1/commands/load",
            json={"load_w": 200e6},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 200
        row = (await entries(client, admin_tokens, endpoint="/api/v1/commands"))[0]
        assert row["outcome"] == "refused: E-Stop active"

    async def test_the_query_string_is_the_detail_of_a_read(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.get(
            "/api/v1/users", params={"active": "true"}, headers=bearer(admin_tokens)
        )
        row = (await entries(client, admin_tokens, endpoint="/api/v1/users"))[0]
        assert row["detail"] == "active=true"


class TestActors:
    """Rows an investigator filters by user must name the user whenever it is known."""

    async def test_a_sign_out_with_the_cookie_alone_names_the_user(
        self, app: FastAPI, admin_tokens: dict[str, str]
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="https://console.test"
        ) as browser:
            signed_in = await browser.post(
                "/auth/login",
                json={"username": "operator1", "password": "operator_password"},
            )
            assert signed_in.status_code == 200
            assert (await browser.post("/auth/logout")).status_code == 200
            row = (await entries(browser, admin_tokens, endpoint="/auth/logout"))[0]
        assert row["username"] == "operator1"
        assert row["role"] == "operator"
        assert row["outcome"] == "signed out"

    async def test_a_sign_out_without_a_session_names_nobody(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.post("/auth/logout", json={"refresh_token": "not-a-jwt"})
        row = (await entries(client, admin_tokens, endpoint="/auth/logout"))[0]
        assert row["user_id"] is None
        assert row["outcome"] == "no open session"

    async def test_a_detected_token_reuse_names_the_user(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        admin_tokens: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "refresh_reuse_grace_s", 0.0)
        body = {"refresh_token": viewer_tokens["refresh"]}
        assert (await client.post("/auth/refresh", json=body)).status_code == 200
        assert (await client.post("/auth/refresh", json=body)).status_code == 401
        row = (await entries(client, admin_tokens, endpoint="/auth/refresh"))[0]
        assert row["outcome"] == "refused: token reuse, session closed"
        assert row["username"] == "viewer1"

    async def test_a_refresh_with_a_forged_token_names_nobody(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        forged = issue_refresh_token(1, "admin", "never-stored").token
        refused = await client.post("/auth/refresh", json={"refresh_token": forged})
        assert refused.status_code == 401
        row = (await entries(client, admin_tokens, endpoint="/auth/refresh"))[0]
        assert row["user_id"] is None


class TestReading:
    async def test_newest_first_with_paging(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        for _ in range(3):
            await client.post(
                "/auth/login", json={"username": "ghost", "password": "wrong_password"}
            )
        response = await client.get(
            "/api/v1/audit",
            params={"endpoint": "/auth/login", "limit": 2, "offset": 1},
            headers=bearer(admin_tokens),
        )
        page = response.json()
        assert page["total"] == 3
        assert len(page["items"]) == 2
        stamps = [(item["timestamp_ms"], item["id"]) for item in page["items"]]
        assert stamps == sorted(stamps, reverse=True)

    async def test_filters_combine(
        self,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        operator_tokens: dict[str, str],
    ) -> None:
        await client.post(
            "/api/v1/commands/mode",
            json={"mode": "manual"},
            headers=bearer(operator_tokens),
        )
        await client.post(
            "/auth/login", json={"username": "ghost", "password": "wrong_password"}
        )
        by_user = await entries(client, admin_tokens, username="operator1")
        assert {row["endpoint"] for row in by_user} == {"/api/v1/commands/mode"}
        by_method = await entries(client, admin_tokens, method="POST", min_status=400)
        assert [row["endpoint"] for row in by_method] == ["/auth/login"]
        operator_id = by_user[0]["user_id"]
        assert await entries(client, admin_tokens, user_id=operator_id) == by_user

    async def test_the_endpoint_filter_is_a_literal_prefix(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.get("/api/v1/users", headers=bearer(admin_tokens))
        assert await entries(client, admin_tokens, endpoint="/api/v1/u_ers") == []
        assert await entries(client, admin_tokens, endpoint="/api/v1/%") == []
        assert await entries(client, admin_tokens, endpoint="/api/v1/us") != []

    async def test_a_time_window_selects_rows(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        await client.get("/api/v1/users", headers=bearer(admin_tokens))
        row = (await entries(client, admin_tokens))[0]
        stamp = row["timestamp_ms"]
        assert await entries(client, admin_tokens, from_ms=stamp + 60_000) == []
        within = await entries(client, admin_tokens, from_ms=stamp, to_ms=stamp)
        assert row["id"] in {item["id"] for item in within}

    async def test_an_inverted_window_is_422(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/audit",
            params={"from_ms": 10, "to_ms": 5},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid_range"

    async def test_an_unknown_method_filter_is_refused(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/audit", params={"method": "TRACE"}, headers=bearer(admin_tokens)
        )
        assert response.status_code == 422


class TestWriteFailure:
    async def test_a_record_that_cannot_be_stored_goes_to_the_error_log(
        self, app: FastAPI, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def broken_database() -> AsyncGenerator[Any]:
            raise ConnectionError("database is down")
            yield None

        app.dependency_overrides[get_db] = broken_database
        entry = AuditLog(
            user_id=4,
            username="operator1",
            role="operator",
            ip_address="10.0.0.9",
            method="POST",
            endpoint="/api/v1/commands/mode",
            request_body_hash=None,
            response_status=200,
            duration_ms=3,
            timestamp_ms=1_700_000_000_000,
            detail=None,
            outcome="accepted",
        )
        with caplog.at_level(logging.ERROR, logger="api_gateway.audit"):
            await write_audit_entry(app, entry)
        message = caplog.records[-1].getMessage()
        assert "Audit record NOT stored" in message
        assert "/api/v1/commands/mode" in message
        assert "operator1" in message
