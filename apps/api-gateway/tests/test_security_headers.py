"""Headers the gateway sets itself, so a host-run gateway behind the Vite proxy has them too.

Responses that carry tokens or personal data must not be stored by any cache (RFC 6749
§5.1 for the token responses); every response forbids content sniffing.
"""

from __future__ import annotations

import pytest
from gateway_fakes import bearer
from httpx import AsyncClient


def not_stored(headers: object) -> bool:
    assert hasattr(headers, "get")
    return (
        headers.get("cache-control") == "no-store"
        and headers.get("pragma") == "no-cache"
    )


class TestNoStore:
    async def test_a_sign_in_answer_is_not_stored(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        assert response.status_code == 200
        assert not_stored(response.headers)

    async def test_a_refreshed_pair_is_not_stored(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/auth/refresh", json={"refresh_token": viewer_tokens["refresh"]}
        )
        assert response.status_code == 200
        assert not_stored(response.headers)

    async def test_a_refused_sign_in_is_not_stored_either(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "wrong_password"}
        )
        assert response.status_code == 401
        assert not_stored(response.headers)

    @pytest.mark.parametrize("path", ["/api/v1/users", "/api/v1/audit", "/auth/me"])
    async def test_accounts_and_the_audit_log_are_not_stored(
        self, client: AsyncClient, admin_tokens: dict[str, str], path: str
    ) -> None:
        response = await client.get(path, headers=bearer(admin_tokens))
        assert response.status_code == 200
        assert not_stored(response.headers)

    async def test_plant_telemetry_keeps_its_own_caching(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/status", headers=bearer(viewer_tokens))
        assert response.status_code == 200
        assert "cache-control" not in response.headers


class TestNoSniff:
    @pytest.mark.parametrize("path", ["/health", "/api/v1/status", "/auth/me"])
    async def test_every_answer_forbids_sniffing(
        self, client: AsyncClient, path: str
    ) -> None:
        response = await client.get(path)
        assert response.headers["x-content-type-options"] == "nosniff"
