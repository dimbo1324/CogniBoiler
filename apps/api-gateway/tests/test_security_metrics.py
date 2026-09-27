"""Security events the gateway counts, so a dashboard can alert on them."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from api_gateway.audit import write_audit_entry
from api_gateway.auth.throttle import LoginThrottle, ThrottlePolicy
from api_gateway.config import settings
from api_gateway.dependencies import get_db
from api_gateway.models.user import AuditLog
from fastapi import FastAPI
from httpx import AsyncClient
from prometheus_client import REGISTRY

LOGIN_FAILURES = "gateway_login_failures_total"
REFRESH_REJECTIONS = "gateway_refresh_rejections_total"
AUDIT_WRITE_FAILURES = "gateway_audit_write_failures_total"


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def entry() -> AuditLog:
    return AuditLog(
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


class TestAuditWrites:
    async def test_a_stored_record_counts_no_failure(self, app: FastAPI) -> None:
        before = sample(AUDIT_WRITE_FAILURES)
        await write_audit_entry(app, entry())
        assert sample(AUDIT_WRITE_FAILURES) == before

    async def test_a_record_that_cannot_be_stored_is_counted(
        self, app: FastAPI
    ) -> None:
        async def broken_database() -> AsyncGenerator[Any]:
            raise ConnectionError("database is down")
            yield None

        app.dependency_overrides[get_db] = broken_database
        before = sample(AUDIT_WRITE_FAILURES)
        await write_audit_entry(app, entry())
        assert sample(AUDIT_WRITE_FAILURES) == before + 1

    async def test_a_cancelled_write_is_logged_counted_and_still_cancels(
        self, app: FastAPI, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def cancelled_database() -> AsyncGenerator[Any]:
            raise asyncio.CancelledError
            yield None

        app.dependency_overrides[get_db] = cancelled_database
        before = sample(AUDIT_WRITE_FAILURES)
        with (
            caplog.at_level(logging.ERROR, logger="api_gateway.audit"),
            pytest.raises(asyncio.CancelledError),
        ):
            await write_audit_entry(app, entry())
        assert sample(AUDIT_WRITE_FAILURES) == before + 1
        assert "Audit record NOT stored" in caplog.text
        assert "/api/v1/commands/mode" in caplog.text


class TestSecurityCounters:
    async def test_failed_sign_ins_are_counted_by_reason(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        app.state.login_throttle = LoginThrottle(
            per_account=ThrottlePolicy(1, 60.0), per_client=ThrottlePolicy(100, 60.0)
        )
        invalid = sample(LOGIN_FAILURES, reason="invalid_credentials")
        throttled = sample(LOGIN_FAILURES, reason="throttled")
        wrong = {"username": "viewer1", "password": "wrong_password"}
        assert (await client.post("/auth/login", json=wrong)).status_code == 401
        assert (await client.post("/auth/login", json=wrong)).status_code == 429
        assert sample(LOGIN_FAILURES, reason="invalid_credentials") == invalid + 1
        assert sample(LOGIN_FAILURES, reason="throttled") == throttled + 1

    async def test_a_successful_sign_in_counts_no_failure(
        self, client: AsyncClient
    ) -> None:
        before = sample(LOGIN_FAILURES, reason="invalid_credentials")
        response = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        assert response.status_code == 200
        assert sample(LOGIN_FAILURES, reason="invalid_credentials") == before

    async def test_a_reused_refresh_token_is_counted(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "refresh_reuse_grace_s", 0.0)
        reused = sample(REFRESH_REJECTIONS, code="auth.refresh_reused")
        body = {"refresh_token": viewer_tokens["refresh"]}
        assert (await client.post("/auth/refresh", json=body)).status_code == 200
        assert sample(REFRESH_REJECTIONS, code="auth.refresh_reused") == reused
        assert (await client.post("/auth/refresh", json=body)).status_code == 401
        assert sample(REFRESH_REJECTIONS, code="auth.refresh_reused") == reused + 1

    async def test_a_refresh_without_a_token_is_counted(
        self, client: AsyncClient
    ) -> None:
        missing = sample(REFRESH_REJECTIONS, code="auth.refresh_missing")
        assert (await client.post("/auth/refresh")).status_code == 401
        assert sample(REFRESH_REJECTIONS, code="auth.refresh_missing") == missing + 1
