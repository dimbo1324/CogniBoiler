"""Sign-in sessions end to end: cookies, rotation, reuse, sign-out, throttling, passwords."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from api_gateway.auth.throttle import LoginThrottle, ThrottlePolicy
from api_gateway.config import settings
from api_gateway.dependencies import get_db
from api_gateway.models.user import User
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

NEW_PASSWORD = "Copper-Kettle-Bridge-7"


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


@pytest_asyncio.fixture
async def browser(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """A client on https, so the Secure refresh cookie is kept and sent back."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://console.test"
    ) as c:
        yield c


def tight_throttle(app: FastAPI, failures: int = 2) -> None:
    app.state.login_throttle = LoginThrottle(
        per_account=ThrottlePolicy(failures, 60.0),
        per_client=ThrottlePolicy(100, 60.0),
    )


async def sign_in(
    client: AsyncClient,
    username: str = "operator1",
    password: str = "operator_password",
) -> dict[str, str]:
    response = await client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return {"access": body["access_token"], "refresh": body["refresh_token"]}


class TestSignIn:
    async def test_the_answer_names_the_user_and_the_expiries(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/auth/login",
            json={"username": "engineer1", "password": "engineer_password"},
        )
        body = response.json()
        assert body["username"] == "engineer1"
        assert body["role"] == "engineer"
        assert body["expires_in"] == settings.jwt_access_token_expire_minutes * 60
        assert body["session_expires_at_ms"] > body["access_expires_at_ms"]

    async def test_the_refresh_cookie_is_http_only_strict_and_scoped(
        self, browser: AsyncClient
    ) -> None:
        response = await browser.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        cookie = response.headers["set-cookie"].lower()
        assert cookie.startswith(f"{settings.refresh_cookie_name}=")
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "secure" in cookie
        assert f"path={settings.refresh_cookie_path}" in cookie

    async def test_a_blocked_account_gets_the_same_answer_as_a_wrong_password(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        sessions = app.dependency_overrides[get_db]()
        db = await anext(sessions)
        await db.execute(
            update(User).where(User.username == "viewer1").values(is_active=False)
        )
        await db.commit()
        await sessions.aclose()

        blocked = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        wrong = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "wrong_password"}
        )
        unknown = await client.post(
            "/auth/login", json={"username": "ghost", "password": "wrong_password"}
        )
        assert blocked.status_code == wrong.status_code == unknown.status_code == 401
        assert (
            blocked.json()["detail"]
            == wrong.json()["detail"]
            == unknown.json()["detail"]
        )

    async def test_the_profile_is_the_signed_in_user(self, client: AsyncClient) -> None:
        tokens = await sign_in(client)
        response = await client.get("/auth/me", headers=bearer(tokens))
        assert response.status_code == 200
        profile = response.json()
        assert profile["username"] == "operator1"
        assert profile["role"] == "operator"
        assert profile["session_id"]

    async def test_the_profile_needs_a_token(self, client: AsyncClient) -> None:
        response = await client.get("/auth/me")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["code"] == "auth.token_missing"


class TestRefresh:
    async def test_the_cookie_alone_refreshes_the_session(
        self, browser: AsyncClient
    ) -> None:
        await sign_in(browser)
        first_cookie = browser.cookies.get(settings.refresh_cookie_name)
        response = await browser.post("/auth/refresh")
        assert response.status_code == 200
        assert browser.cookies.get(settings.refresh_cookie_name) != first_cookie

    async def test_nothing_to_refresh_with_is_401(self, client: AsyncClient) -> None:
        response = await client.post("/auth/refresh")
        assert response.status_code == 401
        assert response.json()["code"] == "auth.refresh_missing"

    async def test_a_second_exchange_within_the_grace_is_only_refused(
        self, client: AsyncClient
    ) -> None:
        tokens = await sign_in(client)
        first = await client.post(
            "/auth/refresh", json={"refresh_token": tokens["refresh"]}
        )
        late = await client.post(
            "/auth/refresh", json={"refresh_token": tokens["refresh"]}
        )
        assert late.status_code == 401
        assert late.json()["code"] == "auth.refresh_superseded"
        successor = await client.post(
            "/auth/refresh", json={"refresh_token": first.json()["refresh_token"]}
        )
        assert successor.status_code == 200

    async def test_a_replayed_token_closes_the_whole_session(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "refresh_reuse_grace_s", 0.0)
        tokens = await sign_in(client)
        first = await client.post(
            "/auth/refresh", json={"refresh_token": tokens["refresh"]}
        )
        replay = await client.post(
            "/auth/refresh", json={"refresh_token": tokens["refresh"]}
        )
        assert replay.status_code == 401
        assert replay.json()["code"] == "auth.refresh_reused"
        assert settings.refresh_cookie_name in replay.headers["set-cookie"]
        successor = await client.post(
            "/auth/refresh", json={"refresh_token": first.json()["refresh_token"]}
        )
        assert successor.status_code == 401
        access = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {first.json()['access_token']}"},
        )
        assert access.status_code == 401

    async def test_a_refresh_keeps_the_session_expiry(
        self, client: AsyncClient
    ) -> None:
        login = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        refreshed = await client.post(
            "/auth/refresh", json={"refresh_token": login.json()["refresh_token"]}
        )
        assert (
            refreshed.json()["session_expires_at_ms"]
            == login.json()["session_expires_at_ms"]
        )


class TestSignOut:
    async def test_a_bearer_token_alone_closes_its_session(
        self, client: AsyncClient
    ) -> None:
        tokens = await sign_in(client)
        response = await client.post("/auth/logout", headers=bearer(tokens))
        assert response.status_code == 200
        again = await client.post(
            "/auth/refresh", json={"refresh_token": tokens["refresh"]}
        )
        assert again.status_code == 401

    async def test_signing_out_without_a_session_still_answers_200(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/auth/logout", headers={"Authorization": "Bearer not.a.token"}
        )
        assert response.status_code == 200

    async def test_signing_out_clears_the_cookie(self, browser: AsyncClient) -> None:
        await sign_in(browser)
        response = await browser.post("/auth/logout")
        assert response.status_code == 200
        assert "max-age=0" in response.headers["set-cookie"].lower()
        assert (await browser.post("/auth/refresh")).status_code == 401


class TestThrottle:
    async def test_repeated_failures_are_answered_429_with_retry_after(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        tight_throttle(app)
        for _ in range(2):
            await client.post(
                "/auth/login",
                json={"username": "viewer1", "password": "wrong_password"},
            )
        response = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        assert response.status_code == 429
        assert response.json()["code"] == "auth.too_many_attempts"
        assert int(response.headers["retry-after"]) > 0

    async def test_an_unknown_account_is_throttled_the_same_way(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        tight_throttle(app)
        statuses = [
            (
                await client.post(
                    "/auth/login",
                    json={"username": "ghost", "password": "wrong_password"},
                )
            ).status_code
            for _ in range(3)
        ]
        assert statuses == [401, 401, 429]

    async def test_a_success_clears_the_account_failures(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        tight_throttle(app)
        await client.post(
            "/auth/login", json={"username": "viewer1", "password": "wrong_password"}
        )
        await sign_in(client, "viewer1", "viewer_password")
        await client.post(
            "/auth/login", json={"username": "viewer1", "password": "wrong_password"}
        )
        response = await client.post(
            "/auth/login", json={"username": "viewer1", "password": "viewer_password"}
        )
        assert response.status_code == 200


class TestPasswordChange:
    async def test_a_change_returns_a_fresh_session_and_closes_the_others(
        self, client: AsyncClient
    ) -> None:
        other = await sign_in(client)
        current = await sign_in(client)
        response = await client.post(
            "/auth/password",
            json={
                "current_password": "operator_password",
                "new_password": NEW_PASSWORD,
            },
            headers=bearer(current),
        )
        assert response.status_code == 200
        assert response.json()["refresh_token"] not in (
            other["refresh"],
            current["refresh"],
        )
        for old in (other, current):
            refused = await client.post(
                "/auth/refresh", json={"refresh_token": old["refresh"]}
            )
            assert refused.status_code == 401
        await sign_in(client, "operator1", NEW_PASSWORD)

    async def test_a_wrong_current_password_is_403(self, client: AsyncClient) -> None:
        tokens = await sign_in(client)
        response = await client.post(
            "/auth/password",
            json={"current_password": "not-my-password", "new_password": NEW_PASSWORD},
            headers=bearer(tokens),
        )
        assert response.status_code == 403
        assert response.json()["code"] == "auth.password_mismatch"

    async def test_the_same_password_again_is_refused(
        self, client: AsyncClient
    ) -> None:
        tokens = await sign_in(client, "engineer1", "engineer_password")
        response = await client.post(
            "/auth/password",
            json={
                "current_password": "engineer_password",
                "new_password": "engineer_password",
            },
            headers=bearer(tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "users.password_unchanged"

    async def test_the_policy_applies_to_a_self_service_change(
        self, client: AsyncClient
    ) -> None:
        tokens = await sign_in(client)
        response = await client.post(
            "/auth/password",
            json={
                "current_password": "operator_password",
                "new_password": "operator1-forever",
            },
            headers=bearer(tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "users.password_weak"

    async def test_wrong_current_passwords_count_toward_the_throttle(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        tokens = await sign_in(client)
        tight_throttle(app)
        codes = []
        for _ in range(3):
            response = await client.post(
                "/auth/password",
                json={"current_password": "guess-guess", "new_password": NEW_PASSWORD},
                headers=bearer(tokens),
            )
            codes.append(response.status_code)
        assert codes == [403, 403, 429]
