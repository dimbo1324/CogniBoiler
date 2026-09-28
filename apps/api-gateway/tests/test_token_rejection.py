"""Access tokens the gateway must refuse, and the signing keys it must refuse to start with.

Each refusal is paired with the token that passes, so a check that refused everything
would fail here too.
"""

from __future__ import annotations

import base64
import configparser
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import jwt
import pytest
from api_gateway import dependencies
from api_gateway.auth import jwt_handler
from api_gateway.auth.jwt_handler import (
    AUDIENCE,
    ISSUER,
    issue_access_token,
    validate_signing_keys,
)
from api_gateway.config import Settings, settings
from api_gateway.dependencies import get_db
from api_gateway.models.user import User
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.engine import make_url


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _pem_pair(bits: int = 2048) -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    public = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return private, public


def claims(session_id: str, user_id: int, **overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": "operator",
        "type": "access",
        "jti": "test-jti",
        "sid": session_id,
        "iat": now,
        "exp": now + 600,
        "iss": ISSUER,
        "aud": AUDIENCE,
    }
    payload.update(overrides)
    return {key: value for key, value in payload.items() if value is not None}


def signed(payload: dict[str, Any], private_pem: str | None = None) -> str:
    return jwt.encode(
        payload, private_pem or settings.jwt_private_key, algorithm="RS256"
    )


def unsigned(header: dict[str, Any], payload: dict[str, Any], key: bytes) -> str:
    """A token built by hand, for algorithms PyJWT refuses to produce with these keys."""
    head = _b64(json.dumps(header).encode())
    body = _b64(json.dumps(payload).encode())
    if header["alg"] == "none":
        return f"{head}.{body}."
    signature = hmac.new(key, f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{_b64(signature)}"


async def session_of(client: AsyncClient, tokens: dict[str, str]) -> tuple[str, int]:
    profile = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {tokens['access']}"}
    )
    assert profile.status_code == 200, profile.text
    body = profile.json()
    return str(body["session_id"]), int(body["id"])


async def status_with(client: AsyncClient, token: str) -> tuple[int, str]:
    response = await client.get(
        "/api/v1/status", headers={"Authorization": f"Bearer {token}"}
    )
    return response.status_code, str(response.json().get("code", ""))


class TestAccessTokens:
    async def test_a_well_formed_token_of_an_open_session_passes(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        status, _ = await status_with(client, signed(claims(session_id, user_id)))
        assert status == 200

    async def test_an_expired_token_is_401_token_expired(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        expired = issue_access_token(
            user_id,
            "operator",
            session_id,
            not_after_ms=int(time.time()) * 1000 - 60_000,
        )
        assert await status_with(client, expired.token) == (401, "auth.token_expired")

    @pytest.mark.parametrize(
        "change",
        [
            {"sid": None},
            {"sid": ""},
            {"sub": "abc"},
            {"iss": None},
            {"iss": "someone-else"},
            {"aud": None},
            {"aud": "another-service"},
            {"type": "refresh"},
        ],
        ids=[
            "no-sid",
            "empty-sid",
            "non-numeric-sub",
            "no-iss",
            "foreign-iss",
            "no-aud",
            "foreign-aud",
            "refresh-as-access",
        ],
    )
    async def test_a_token_with_a_bad_claim_is_401_token_invalid(
        self,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        change: dict[str, Any],
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        token = signed(claims(session_id, user_id, **change))
        assert await status_with(client, token) == (401, "auth.token_invalid")

    async def test_a_token_signed_with_another_key_is_401(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        foreign_private, _ = _pem_pair()
        token = signed(claims(session_id, user_id), foreign_private)
        assert await status_with(client, token) == (401, "auth.token_invalid")

    async def test_an_hs256_token_keyed_with_the_public_key_is_401(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        token = unsigned(
            {"alg": "HS256", "typ": "JWT"},
            claims(session_id, user_id),
            settings.jwt_public_key.encode(),
        )
        assert await status_with(client, token) == (401, "auth.token_invalid")

    async def test_an_unsigned_token_is_401(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        session_id, user_id = await session_of(client, operator_tokens)
        token = unsigned(
            {"alg": "none", "typ": "JWT"}, claims(session_id, user_id), b""
        )
        assert await status_with(client, token) == (401, "auth.token_invalid")

    async def test_a_refresh_token_is_not_a_bearer_token(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        assert await status_with(client, operator_tokens["refresh"]) == (
            401,
            "auth.token_invalid",
        )

    async def test_a_token_of_a_deleted_user_is_401_session_invalid(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        _, user_id = await session_of(client, operator_tokens)
        sessions = app.dependency_overrides[get_db]()
        db = await anext(sessions)
        await db.execute(delete(User).where(User.id == user_id))
        await db.commit()
        await sessions.aclose()
        assert await status_with(client, operator_tokens["access"]) == (
            401,
            "auth.session_invalid",
        )

    async def test_a_session_closed_by_an_admin_ends_the_access_token(
        self,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        _, user_id = await session_of(client, operator_tokens)
        closed = await client.post(
            f"/api/v1/users/{user_id}/revoke-sessions",
            headers={"Authorization": f"Bearer {admin_tokens['access']}"},
        )
        assert closed.status_code == 200
        me = await client.get(
            "/auth/me", headers={"Authorization": f"Bearer {operator_tokens['access']}"}
        )
        assert me.status_code == 401
        assert me.json()["code"] == "auth.session_invalid"


class TestKeysAreRequired:
    def test_signing_without_a_private_key_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "jwt_private_key", "")
        with pytest.raises(RuntimeError, match="jwt_private_key"):
            issue_access_token(1, "viewer", "session")

    def test_verifying_without_a_public_key_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = issue_access_token(1, "viewer", "session").token
        monkeypatch.setattr(settings, "jwt_public_key", "")
        with pytest.raises(RuntimeError, match="jwt_public_key"):
            jwt_handler.decode_access_token(token)


class TestKeysAreParsedOnce:
    def test_issuing_twice_parses_the_private_key_once(self) -> None:
        issue_access_token(1, "viewer", "session")
        before = jwt_handler.signing_key.cache_info()
        issue_access_token(2, "viewer", "session")
        issue_access_token(3, "viewer", "session")
        after = jwt_handler.signing_key.cache_info()
        assert after.misses == before.misses
        assert after.hits >= before.hits + 2

    def test_verifying_twice_parses_the_public_key_once(self) -> None:
        token = issue_access_token(1, "viewer", "session").token
        jwt_handler.decode_access_token(token)
        before = jwt_handler.verifying_key.cache_info()
        jwt_handler.decode_access_token(token)
        after = jwt_handler.verifying_key.cache_info()
        assert after.misses == before.misses

    def test_a_swapped_key_is_used_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        private, public = _pem_pair()
        monkeypatch.setattr(settings, "jwt_private_key", private)
        monkeypatch.setattr(settings, "jwt_public_key", public)
        token = issue_access_token(1, "viewer", "session").token
        assert jwt_handler.decode_access_token(token)["sub"] == "1"


class TestStartUpValidation:
    def test_the_configured_pair_is_accepted(self) -> None:
        validate_signing_keys()

    def test_a_missing_key_stops_start_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "jwt_public_key", "")
        with pytest.raises(RuntimeError, match="jwt_public_key"):
            validate_signing_keys()

    def test_a_mismatched_pair_stops_start_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, other_public = _pem_pair()
        monkeypatch.setattr(settings, "jwt_public_key", other_public)
        with pytest.raises(RuntimeError, match="do not match"):
            validate_signing_keys()

    def test_a_short_key_stops_start_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        private, public = _pem_pair(bits=1024)
        monkeypatch.setattr(settings, "jwt_private_key", private)
        monkeypatch.setattr(settings, "jwt_public_key", public)
        with pytest.raises(RuntimeError, match="2048"):
            validate_signing_keys()

    def test_text_that_is_not_a_key_stops_start_up_without_being_echoed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "jwt_private_key", "not-a-key-at-all-42")
        with pytest.raises(RuntimeError) as raised:
            validate_signing_keys()
        assert "not-a-key-at-all-42" not in str(raised.value)
        assert raised.value.__cause__ is None


class TestSettingsArePinned:
    def test_the_algorithm_cannot_be_configured_away_from_rs256(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, jwt_algorithm="HS256")  # type: ignore[arg-type]
        assert Settings(_env_file=None).jwt_algorithm == "RS256"

    @pytest.mark.parametrize(
        "field",
        [
            {"jwt_access_token_expire_minutes": 0},
            {"jwt_access_token_expire_minutes": 61},
            {"refresh_reuse_grace_s": -1.0},
            {"refresh_reuse_grace_s": 31.0},
            {"login_max_failures_per_account": 0},
            {"login_max_failures_per_client": 0},
            {"login_failure_window_s": 0.0},
            {"login_max_concurrent_hashes": 0},
        ],
    )
    def test_values_that_would_disable_a_defence_are_refused(
        self, field: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **field)

    def test_the_defaults_are_within_the_bounds(self) -> None:
        defaults = Settings(_env_file=None)
        assert 1 <= defaults.jwt_access_token_expire_minutes <= 60
        assert 0 <= defaults.refresh_reuse_grace_s <= 30


class TestNoWorkingDefaults:
    def test_there_is_no_default_database(self) -> None:
        assert Settings(_env_file=None).database_url == ""

    def test_without_a_database_url_the_first_session_says_what_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "database_url", "")
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            dependencies.session_factory()

    def test_the_engine_is_made_on_first_use_once_per_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///first.db")
        first = dependencies.session_factory()
        assert dependencies.session_factory() is first
        monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///other.db")
        other = dependencies.session_factory()
        assert other is not first
        assert make_url(str(other.kw["bind"].url)).database == "other.db"

    def test_no_browser_origin_is_trusted_by_default(self) -> None:
        assert Settings(_env_file=None).cors_allowed_origins == []

    def test_an_origin_can_still_be_configured(self) -> None:
        configured = Settings(
            _env_file=None, cors_allowed_origins=["http://127.0.0.1:5173"]
        )
        assert configured.cors_allowed_origins == ["http://127.0.0.1:5173"]

    def test_alembic_ini_holds_no_database_url_that_could_connect(self) -> None:
        ini = Path(__file__).resolve().parents[1] / "alembic.ini"
        parser = configparser.ConfigParser()
        parser.read(ini, encoding="utf-8")
        url = parser.get("alembic", "sqlalchemy.url")
        assert "@" not in url
        assert "set-DATABASE_URL" in url
