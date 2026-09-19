"""User administration: listing, creation, roles, blocking, password resets, sign-outs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from api_gateway import accounts
from api_gateway.auth.identity import CurrentUser, load_account
from api_gateway.dependencies import get_db
from api_gateway.models.user import Role
from api_gateway.problems import ProblemError
from api_gateway.schemas.users import UserUpdateRequest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

NEW_PASSWORD = "Harbour-Lantern-42"


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


@asynccontextmanager
async def database(app: FastAPI) -> AsyncIterator[AsyncSession]:
    sessions = app.dependency_overrides[get_db]()
    session = await anext(sessions)
    try:
        yield session
    finally:
        await sessions.aclose()


async def user_id(app: FastAPI, username: str) -> int:
    async with database(app) as db:
        account = await load_account(db, username=username)
    assert account is not None
    return account.id


async def login(client: AsyncClient, username: str, password: str) -> int:
    response = await client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    return response.status_code


async def refresh_status(client: AsyncClient, refresh_token: str) -> int:
    response = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    return response.status_code


class TestListing:
    async def test_accounts_are_ordered_by_username_with_their_roles(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/users", headers=bearer(admin_tokens))
        assert response.status_code == 200
        page = response.json()
        names = [item["username"] for item in page["items"]]
        assert names == sorted(names)
        assert page["total"] == 5
        roles = {item["username"]: item["role"] for item in page["items"]}
        assert roles == {
            "admin1": "admin",
            "engineer1": "engineer",
            "operator1": "operator",
            "testuser": "operator",
            "viewer1": "viewer",
        }

    async def test_open_sessions_are_counted(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/users", headers=bearer(admin_tokens))
        sessions = {
            item["username"]: item["open_sessions"] for item in response.json()["items"]
        }
        assert sessions["admin1"] == 1
        assert sessions["viewer1"] == 0

    async def test_the_active_filter_and_paging_apply(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        blocked = await client.get(
            "/api/v1/users", params={"active": "false"}, headers=bearer(admin_tokens)
        )
        assert blocked.json()["items"] == []
        page = await client.get(
            "/api/v1/users",
            params={"limit": 2, "offset": 1},
            headers=bearer(admin_tokens),
        )
        body = page.json()
        assert [item["username"] for item in body["items"]] == [
            "engineer1",
            "operator1",
        ]
        assert (body["total"], body["limit"], body["offset"]) == (5, 2, 1)

    async def test_an_engineer_may_not_list_accounts(
        self, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/users", headers=bearer(engineer_tokens))
        assert response.status_code == 403
        assert response.json()["code"] == "auth.forbidden"
        assert response.json()["required_role"] == "admin"

    async def test_an_unknown_user_is_404(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/users/999", headers=bearer(admin_tokens))
        assert response.status_code == 404
        assert response.json()["code"] == "users.not_found"


class TestCreation:
    async def test_a_new_account_is_active_with_its_role_and_can_sign_in(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={
                "username": "shift.lead",
                "password": NEW_PASSWORD,
                "role": "engineer",
            },
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 201
        created = response.json()
        assert created["username"] == "shift.lead"
        assert created["role"] == "engineer"
        assert created["is_active"] is True
        assert created["last_login_at_ms"] is None
        assert await login(client, "shift.lead", NEW_PASSWORD) == 200

    async def test_a_username_is_unique_ignoring_case(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={"username": "Operator1", "password": NEW_PASSWORD, "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 409
        assert response.json()["code"] == "users.username_taken"

    @pytest.mark.parametrize(
        ("password", "reason"),
        [
            ("my-shift.lead-pass", "contains the username"),
            ("aaaabbbbaaaa", "too few different characters"),
        ],
    )
    async def test_weak_passwords_are_refused(
        self,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        password: str,
        reason: str,
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={"username": "shift.lead", "password": password, "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422, reason
        assert response.json()["code"] == "users.password_weak"

    async def test_a_short_password_fails_validation_without_being_echoed(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={"username": "shift.lead", "password": "Short-1", "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid"
        assert "Short-1" not in response.text

    @pytest.mark.parametrize("username", ["ab", "has space", "semi;colon", "x" * 65])
    async def test_usernames_outside_the_pattern_are_refused(
        self, client: AsyncClient, admin_tokens: dict[str, str], username: str
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={"username": username, "password": NEW_PASSWORD, "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422

    async def test_an_unknown_role_is_refused(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users",
            json={"username": "shift.lead", "password": NEW_PASSWORD, "role": "root"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422

    async def test_a_role_missing_from_the_database_is_a_conflict(
        self, app: FastAPI, admin_tokens: dict[str, str], client: AsyncClient
    ) -> None:
        async with database(app) as db:
            await db.execute(delete(Role).where(Role.name == "viewer"))
            await db.commit()
        response = await client.post(
            "/api/v1/users",
            json={"username": "shift.lead", "password": NEW_PASSWORD, "role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 409
        assert response.json()["code"] == "users.role_missing"


class TestRoleAndBlocking:
    async def test_a_role_change_takes_effect_and_closes_the_sessions(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        operator_tokens: dict[str, str],
    ) -> None:
        target = await user_id(app, "operator1")
        response = await client.patch(
            f"/api/v1/users/{target}",
            json={"role": "engineer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 200
        assert response.json()["role"] == "engineer"
        assert response.json()["open_sessions"] == 0
        assert await refresh_status(client, operator_tokens["refresh"]) == 401
        status = await client.get("/api/v1/status", headers=bearer(operator_tokens))
        assert status.status_code == 401

    async def test_blocking_refuses_sign_in_and_the_existing_token(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        viewer_tokens: dict[str, str],
    ) -> None:
        target = await user_id(app, "viewer1")
        response = await client.patch(
            f"/api/v1/users/{target}",
            json={"is_active": False},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 200
        assert response.json()["is_active"] is False
        assert await login(client, "viewer1", "viewer_password") == 401
        status = await client.get("/api/v1/status", headers=bearer(viewer_tokens))
        assert status.status_code == 401

        unblocked = await client.patch(
            f"/api/v1/users/{target}",
            json={"is_active": True},
            headers=bearer(admin_tokens),
        )
        assert unblocked.json()["is_active"] is True
        assert await login(client, "viewer1", "viewer_password") == 200

    async def test_an_unchanged_role_keeps_the_sessions(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        viewer_tokens: dict[str, str],
    ) -> None:
        target = await user_id(app, "viewer1")
        response = await client.patch(
            f"/api/v1/users/{target}",
            json={"role": "viewer", "is_active": True},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 200
        assert response.json()["open_sessions"] == 1
        assert await refresh_status(client, viewer_tokens["refresh"]) == 200

    async def test_an_empty_change_fails_validation(
        self, app: FastAPI, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        target = await user_id(app, "viewer1")
        response = await client.patch(
            f"/api/v1/users/{target}", json={}, headers=bearer(admin_tokens)
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "change", [{"role": "operator"}, {"is_active": False}], ids=["demote", "block"]
    )
    async def test_an_administrator_cannot_change_their_own_account(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        change: dict[str, object],
    ) -> None:
        own = await user_id(app, "admin1")
        response = await client.patch(
            f"/api/v1/users/{own}", json=change, headers=bearer(admin_tokens)
        )
        assert response.status_code == 409
        assert response.json()["code"] == "users.self_change"

    async def test_changing_an_unknown_user_is_404(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.patch(
            "/api/v1/users/999", json={"role": "viewer"}, headers=bearer(admin_tokens)
        )
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "change",
        [UserUpdateRequest(role="engineer"), UserUpdateRequest(is_active=False)],
        ids=["demote", "block"],
    )
    async def test_the_last_active_administrator_is_kept(
        self, app: FastAPI, change: UserUpdateRequest
    ) -> None:
        admin = await user_id(app, "admin1")
        other_actor = CurrentUser(
            id=10_000,
            username="elsewhere",
            role="admin",
            session_id="s",
            token_expires_at_ms=0,
        )
        async with database(app) as db:
            with pytest.raises(ProblemError) as refused:
                await accounts.update_user(db, other_actor, admin, change)
        assert refused.value.code == "users.last_admin"

    async def test_a_second_administrator_may_be_demoted(
        self, app: FastAPI, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        created = await client.post(
            "/api/v1/users",
            json={"username": "admin2", "password": NEW_PASSWORD, "role": "admin"},
            headers=bearer(admin_tokens),
        )
        response = await client.patch(
            f"/api/v1/users/{created.json()['id']}",
            json={"role": "viewer"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 200
        assert response.json()["role"] == "viewer"


class TestPasswordsAndSessions:
    async def test_a_reset_replaces_the_password_and_closes_the_sessions(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        operator_tokens: dict[str, str],
    ) -> None:
        target = await user_id(app, "operator1")
        response = await client.post(
            f"/api/v1/users/{target}/password",
            json={"new_password": NEW_PASSWORD},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 200
        assert await login(client, "operator1", "operator_password") == 401
        assert await login(client, "operator1", NEW_PASSWORD) == 200
        assert await refresh_status(client, operator_tokens["refresh"]) == 401

    async def test_a_reset_obeys_the_password_policy(
        self, app: FastAPI, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        target = await user_id(app, "operator1")
        response = await client.post(
            f"/api/v1/users/{target}/password",
            json={"new_password": "operator1-password"},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "users.password_weak"

    async def test_a_reset_of_an_unknown_user_is_404(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users/999/password",
            json={"new_password": NEW_PASSWORD},
            headers=bearer(admin_tokens),
        )
        assert response.status_code == 404

    async def test_signing_a_user_out_everywhere(
        self,
        app: FastAPI,
        client: AsyncClient,
        admin_tokens: dict[str, str],
        engineer_tokens: dict[str, str],
    ) -> None:
        target = await user_id(app, "engineer1")
        response = await client.post(
            f"/api/v1/users/{target}/revoke-sessions", headers=bearer(admin_tokens)
        )
        assert response.status_code == 200
        assert response.json() == {"user_id": target, "revoked_tokens": 1}
        assert await refresh_status(client, engineer_tokens["refresh"]) == 401

    async def test_signing_out_an_unknown_user_is_404(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/users/999/revoke-sessions", headers=bearer(admin_tokens)
        )
        assert response.status_code == 404

    async def test_a_sign_in_is_recorded_on_the_account(
        self, app: FastAPI, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        assert await login(client, "viewer1", "viewer_password") == 200
        target = await user_id(app, "viewer1")
        response = await client.get(
            f"/api/v1/users/{target}", headers=bearer(admin_tokens)
        )
        assert response.json()["last_login_at_ms"] > 1


class TestPasswordPolicy:
    def test_a_long_varied_password_passes(self) -> None:
        accounts.check_password_policy("operator1", NEW_PASSWORD)

    def test_the_username_is_found_ignoring_case(self) -> None:
        with pytest.raises(ProblemError) as refused:
            accounts.check_password_policy("Operator1", "xx-OPERATOR1-xx")
        assert refused.value.status == 422
