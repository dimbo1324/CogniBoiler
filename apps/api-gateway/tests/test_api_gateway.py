"""
Integration tests for the API Gateway.

Tests cover:
  1. Password hashing and verification  (auth/password.py)
  2. JWT token creation and decoding    (auth/jwt_handler.py)
  3. RBAC role level hierarchy          (auth/rbac.py)
  4. HTTP endpoints via AsyncClient     (routers/*)

No real database or MQTT broker required — all external dependencies
are stubbed at the module level or overridden via monkeypatch.

FastAPI test pattern:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import jwt as pyjwt
import pytest
import pytest_asyncio
from alert_manager.models import Base as AlertBase
from api_gateway.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)
from api_gateway.auth.password import hash_password, verify_password
from api_gateway.auth.rbac import _role_level
from api_gateway.dependencies import get_db
from api_gateway.main import create_app
from api_gateway.models.user import Base, Role, User, UserRole
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# ─── App fixture ──────────────────────────────────────────────────────────────


class FakePhysicsClient:
    async def close(self) -> None:
        return None

    async def get_system_state(self) -> pb2.SystemStateMsg:
        return pb2.SystemStateMsg(
            boiler=pb2.BoilerStateMsg(
                pressure_pa=140.0e5,
                water_level_m=4.8,
                water_temp_k=611.0,
                flue_gas_temp_k=1200.0,
                internal_energy_j=2.5e12,
                timestamp_ms=1710000000000,
                quality=pb2.SensorQuality.GOOD,
            ),
            turbine=pb2.TurbineStateMsg(
                electrical_power_w=200.0e6,
                shaft_power_w=205.0e6,
                enthalpy_in_j_kg=3_400_000.0,
                enthalpy_out_j_kg=2_200_000.0,
                exhaust_pressure_pa=7_000.0,
                steam_flow_kg_s=150.0,
                timestamp_ms=1710000000000,
                steam_temp_in_k=825.65,
            ),
            actuators=pb2.ActuatorStateMsg(
                fuel_valve_command=0.5,
                fuel_valve_position=0.5,
                feedwater_valve_command=0.5,
                feedwater_valve_position=0.5,
                steam_valve_command=0.5,
                steam_valve_position=0.5,
            ),
        )


class FakePLCClient:
    async def close(self) -> None:
        return None

    async def send_command(self, command: pb2.ControlCommandMsg) -> pb2.CommandAck:
        return pb2.CommandAck(accepted=True, reason="", timestamp_ms=1710000000000)

    async def update_setpoints(self, setpoints: pb2.SetpointsMsg) -> pb2.CommandAck:
        return pb2.CommandAck(accepted=True, reason="", timestamp_ms=1710000000000)

    async def reset_emergency_stop(self, operator_id: str) -> pb2.CommandAck:
        return pb2.CommandAck(accepted=True, reason="", timestamp_ms=1710000000000)


class FakeHistorianClient:
    def close(self) -> None:
        return None

    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, object]]:
        return []


@pytest_asyncio.fixture
async def app() -> AsyncGenerator[FastAPI]:
    """Fresh FastAPI application with DB override and fake upstream clients."""
    application = create_app()
    application.state.physics_client = FakePhysicsClient()
    application.state.plc_client = FakePLCClient()
    application.state.historian_client = FakeHistorianClient()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(AlertBase.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator:
        async with session_factory() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db

    async with session_factory() as session:
        roles = {
            "viewer": Role(name="viewer", description="viewer"),
            "operator": Role(name="operator", description="operator"),
            "engineer": Role(name="engineer", description="engineer"),
            "admin": Role(name="admin", description="admin"),
        }
        session.add_all(roles.values())
        await session.flush()

        users = {
            "testuser": User(
                username="testuser",
                hashed_password=hash_password("test_password"),
                is_active=True,
                created_at_ms=1,
            ),
            "viewer1": User(
                username="viewer1",
                hashed_password=hash_password("viewer_password"),
                is_active=True,
                created_at_ms=1,
            ),
            "operator1": User(
                username="operator1",
                hashed_password=hash_password("operator_password"),
                is_active=True,
                created_at_ms=1,
            ),
            "engineer1": User(
                username="engineer1",
                hashed_password=hash_password("engineer_password"),
                is_active=True,
                created_at_ms=1,
            ),
            "admin1": User(
                username="admin1",
                hashed_password=hash_password("admin_password"),
                is_active=True,
                created_at_ms=1,
            ),
        }
        session.add_all(users.values())
        await session.flush()

        session.add_all(
            [
                UserRole(
                    user_id=users["testuser"].id,
                    role_id=roles["operator"].id,
                    granted_at_ms=1,
                ),
                UserRole(
                    user_id=users["viewer1"].id,
                    role_id=roles["viewer"].id,
                    granted_at_ms=1,
                ),
                UserRole(
                    user_id=users["operator1"].id,
                    role_id=roles["operator"].id,
                    granted_at_ms=1,
                ),
                UserRole(
                    user_id=users["engineer1"].id,
                    role_id=roles["engineer"].id,
                    granted_at_ms=1,
                ),
                UserRole(
                    user_id=users["admin1"].id,
                    role_id=roles["admin"].id,
                    granted_at_ms=1,
                ),
            ]
        )
        await session.commit()

    yield application

    application.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """
    Async HTTP client wired directly to the FastAPI app via ASGI transport.
    No real network socket is opened — requests go through the ASGI interface.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ─── Valid tokens fixtures ────────────────────────────────────────────────────


@pytest.fixture
def viewer_tokens() -> dict[str, str]:
    """JWT token pair for a viewer user (lowest privilege)."""
    return {
        "access": create_access_token(user_id=10, role="viewer"),
        "refresh": create_refresh_token(user_id=10, role="viewer"),
    }


@pytest.fixture
def operator_tokens() -> dict[str, str]:
    """JWT token pair for an operator user."""
    return {
        "access": create_access_token(user_id=20, role="operator"),
        "refresh": create_refresh_token(user_id=20, role="operator"),
    }


@pytest.fixture
def engineer_tokens() -> dict[str, str]:
    """JWT token pair for an engineer user."""
    return {
        "access": create_access_token(user_id=30, role="engineer"),
        "refresh": create_refresh_token(user_id=30, role="engineer"),
    }


@pytest.fixture
def admin_tokens() -> dict[str, str]:
    """JWT token pair for an admin user."""
    return {
        "access": create_access_token(user_id=40, role="admin"),
        "refresh": create_refresh_token(user_id=40, role="admin"),
    }


# ─── 1. Password tests ────────────────────────────────────────────────────────


class TestPassword:
    def test_hash_returns_string(self) -> None:
        result = hash_password("mysecret")
        assert isinstance(result, str)

    def test_hash_starts_with_argon2id(self) -> None:
        result = hash_password("mysecret")
        assert result.startswith("$argon2id$")

    def test_hash_is_not_plaintext(self) -> None:
        result = hash_password("mysecret")
        assert "mysecret" not in result

    def test_two_hashes_of_same_password_differ(self) -> None:
        # Argon2 uses a random salt — same input -> different hash each time
        h1 = hash_password("mysecret")
        h2 = hash_password("mysecret")
        assert h1 != h2

    def test_verify_correct_password_returns_true(self) -> None:
        hashed = hash_password("correct_password")
        assert verify_password("correct_password", hashed) is True

    def test_verify_wrong_password_returns_false(self) -> None:
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_verify_empty_password_returns_false(self) -> None:
        hashed = hash_password("correct_password")
        assert verify_password("", hashed) is False

    def test_verify_malformed_hash_returns_false(self) -> None:
        # Should never raise — always returns False on bad hash
        assert verify_password("password", "not-a-valid-hash") is False

    def test_verify_empty_hash_returns_false(self) -> None:
        assert verify_password("password", "") is False


# ─── 2. JWT handler tests ─────────────────────────────────────────────────────


class TestJWTHandler:
    def test_access_token_is_string(self) -> None:
        token = create_access_token(user_id=1, role="operator")
        assert isinstance(token, str)

    def test_access_token_has_three_parts(self) -> None:
        # JWT format: header.payload.signature
        token = create_access_token(user_id=1, role="operator")
        assert len(token.split(".")) == 3

    def test_decode_access_token_returns_correct_sub(self) -> None:
        token = create_access_token(user_id=42, role="viewer")
        payload = decode_access_token(token)
        assert payload["sub"] == "42"

    def test_decode_access_token_returns_correct_role(self) -> None:
        token = create_access_token(user_id=1, role="engineer")
        payload = decode_access_token(token)
        assert payload["role"] == "engineer"

    def test_decode_access_token_type_is_access(self) -> None:
        token = create_access_token(user_id=1, role="admin")
        payload = decode_access_token(token)
        assert payload["type"] == "access"

    def test_decode_access_token_has_exp(self) -> None:
        token = create_access_token(user_id=1, role="viewer")
        payload = decode_access_token(token)
        assert "exp" in payload
        assert int(str(payload["exp"])) > 0

    def test_refresh_token_type_is_refresh(self) -> None:
        token = create_refresh_token(user_id=1, role="operator")
        payload = decode_refresh_token(token)
        assert payload["type"] == "refresh"

    def test_decode_access_rejects_refresh_token(self) -> None:
        # A refresh token must not be accepted at access-token endpoints
        refresh = create_refresh_token(user_id=1, role="operator")
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_access_token(refresh)

    def test_decode_refresh_rejects_access_token(self) -> None:
        access = create_access_token(user_id=1, role="operator")
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_refresh_token(access)

    def test_tampered_token_raises(self) -> None:
        token = create_access_token(user_id=1, role="operator")
        # Tamper the signature (third part) by replacing several characters
        parts = token.split(".")
        sig = parts[2]
        mid = len(sig) // 2
        tampered_sig = sig[:mid] + ("A" * 8) + sig[mid + 8 :]
        tampered = ".".join([parts[0], parts[1], tampered_sig])
        with pytest.raises(pyjwt.PyJWTError):
            decode_access_token(tampered)

    def test_garbage_token_raises(self) -> None:
        with pytest.raises(pyjwt.PyJWTError):
            decode_access_token("not.a.token")


# ─── 3. RBAC role level tests ─────────────────────────────────────────────────


class TestRBAC:
    def test_viewer_level_is_zero(self) -> None:
        assert _role_level("viewer") == 0

    def test_operator_level_is_one(self) -> None:
        assert _role_level("operator") == 1

    def test_engineer_level_is_two(self) -> None:
        assert _role_level("engineer") == 2

    def test_admin_level_is_three(self) -> None:
        assert _role_level("admin") == 3

    def test_unknown_role_level_is_minus_one(self) -> None:
        assert _role_level("superuser") == -1

    def test_admin_outranks_engineer(self) -> None:
        assert _role_level("admin") > _role_level("engineer")

    def test_engineer_outranks_operator(self) -> None:
        assert _role_level("engineer") > _role_level("operator")

    def test_operator_outranks_viewer(self) -> None:
        assert _role_level("operator") > _role_level("viewer")


# ─── 4. Health endpoint ───────────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_running_status(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_returns_service_name(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert "service" in response.json()

    @pytest.mark.asyncio
    async def test_health_returns_version(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert "version" in response.json()

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client: AsyncClient) -> None:
        # Health endpoint must be accessible without any token
        response = await client.get("/health")
        assert response.status_code != 401
        assert response.status_code != 403


# ─── 5. Auth endpoints ────────────────────────────────────────────────────────


class TestAuthEndpoints:
    @pytest.mark.asyncio
    async def test_login_valid_credentials(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/login",
            json={"username": "testuser", "password": "test_password"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/login",
            json={"username": "testuser", "password": "wrong_password"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_unknown_user_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/login",
            json={"username": "nobody", "password": "password123"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_short_password_returns_422(self, client: AsyncClient) -> None:
        # Pydantic rejects passwords shorter than 8 characters
        response = await client.post(
            "/auth/login",
            json={"username": "testuser", "password": "short"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_refresh_valid_token_returns_new_access(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": viewer_tokens["refresh"]},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_returns_401(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        # Access token must be rejected at the refresh endpoint
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": viewer_tokens["access"]},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_garbage_token_returns_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": "garbage.token.value"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_logout_returns_200(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/auth/logout",
            json={"refresh_token": viewer_tokens["refresh"]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_logout_blacklists_refresh_token(self, client: AsyncClient) -> None:
        # Generate a dedicated token pair for this test to avoid
        # cross-test blacklist pollution from shared fixtures.
        refresh = create_refresh_token(user_id=99, role="viewer")
        await client.post("/auth/logout", json={"refresh_token": refresh})
        # The same token must now be rejected at /auth/refresh
        response = await client.post("/auth/refresh", json={"refresh_token": refresh})
        assert response.status_code == 401


# ─── 6. Status endpoint ───────────────────────────────────────────────────────


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_requires_auth(self, client: AsyncClient) -> None:
        # No token provided — HTTPBearer returns 401
        response = await client.get("/api/v1/status")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_status_viewer_can_access(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/status",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_status_returns_boiler_and_turbine(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/status",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        data = response.json()
        assert "boiler" in data
        assert "turbine" in data

    @pytest.mark.asyncio
    async def test_status_boiler_has_pressure(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/status",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert "pressure_pa" in response.json()["boiler"]

    @pytest.mark.asyncio
    async def test_status_turbine_has_electrical_power(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/status",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert "electrical_power_w" in response.json()["turbine"]

    @pytest.mark.asyncio
    async def test_status_invalid_token_returns_401(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/status",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401


# ─── 7. Commands endpoints ────────────────────────────────────────────────────


class TestCommandsEndpoints:
    @pytest.mark.asyncio
    async def test_valve_requires_auth(self, client: AsyncClient) -> None:
        # No token provided — HTTPBearer returns 401
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 0.5, "feedwater_valve": 0.5, "steam_valve": 0.5},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_valve_viewer_is_forbidden(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        # Viewer cannot send valve commands — minimum role is operator
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 0.5, "feedwater_valve": 0.5, "steam_valve": 0.5},
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_valve_operator_can_send(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 0.5, "feedwater_valve": 0.6, "steam_valve": 0.4},
            headers={"Authorization": f"Bearer {operator_tokens['access']}"},
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True

    @pytest.mark.asyncio
    async def test_valve_admin_can_send(
        self, client: AsyncClient, admin_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 0.8, "feedwater_valve": 0.7, "steam_valve": 0.3},
            headers={"Authorization": f"Bearer {admin_tokens['access']}"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_valve_above_max_returns_422(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        # Pydantic rejects valve > 1.0
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 1.5, "feedwater_valve": 0.5, "steam_valve": 0.5},
            headers={"Authorization": f"Bearer {operator_tokens['access']}"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_valve_below_min_returns_422(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": -0.1, "feedwater_valve": 0.5, "steam_valve": 0.5},
            headers={"Authorization": f"Bearer {operator_tokens['access']}"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_setpoint_operator_is_forbidden(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        # Setpoints require engineer — operator is not enough
        response = await client.post(
            "/api/v1/commands/setpoint",
            json={
                "pressure_pa": 140e5,
                "water_level_m": 4.8,
                "steam_temp_k": 811.0,
            },
            headers={"Authorization": f"Bearer {operator_tokens['access']}"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_setpoint_engineer_can_send(
        self, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/setpoint",
            json={
                "pressure_pa": 140e5,
                "water_level_m": 4.8,
                "steam_temp_k": 811.0,
            },
            headers={"Authorization": f"Bearer {engineer_tokens['access']}"},
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True

    @pytest.mark.asyncio
    async def test_setpoint_pressure_too_high_returns_422(
        self, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        # 200 bar exceeds the safe maximum of 185 bar
        response = await client.post(
            "/api/v1/commands/setpoint",
            json={
                "pressure_pa": 200e5,
                "water_level_m": 4.8,
                "steam_temp_k": 811.0,
            },
            headers={"Authorization": f"Bearer {engineer_tokens['access']}"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_setpoint_pressure_too_low_returns_422(
        self, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/setpoint",
            json={
                "pressure_pa": 10e5,
                "water_level_m": 4.8,
                "steam_temp_k": 811.0,
            },
            headers={"Authorization": f"Bearer {engineer_tokens['access']}"},
        )
        assert response.status_code == 422


# ─── History endpoint ─────────────────────────────────────────────────────────


class RecordedHistorianClient(FakeHistorianClient):
    """Returns one row shaped exactly like a pivoted InfluxDB Flux record."""

    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, object]]:
        from datetime import UTC, datetime

        return [
            {
                "result": "_result",
                "table": 0,
                "_start": datetime(2026, 9, 15, 6, 0, tzinfo=UTC),
                "_stop": datetime(2026, 9, 15, 6, 15, tzinfo=UTC),
                "_time": datetime(2026, 9, 15, 6, 10, tzinfo=UTC),
                "_measurement": measurement,
                "quality": "good",
                "pressure_pa": 140.0e5,
                "water_level_m": 4.8,
                "timestamp_ms": 1_789_452_600_000,
            }
        ]


class FailingHistorianClient(FakeHistorianClient):
    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, object]]:
        raise ConnectionRefusedError("influxdb:8086 refused the connection")


class TestHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_a_real_influx_row_is_returned_without_flux_metadata(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        app.state.historian_client = RecordedHistorianClient()
        response = await client.get(
            "/api/v1/history",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert response.status_code == 200
        point = response.json()["points"][0]
        assert point["timestamp_ms"] == 1_789_452_600_000
        assert point["values"] == {
            "quality": "good",
            "pressure_pa": 140.0e5,
            "water_level_m": 4.8,
        }

    @pytest.mark.asyncio
    async def test_an_unreachable_historian_is_503_not_500(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        app.state.historian_client = FailingHistorianClient()
        response = await client.get(
            "/api/v1/history",
            headers={"Authorization": f"Bearer {viewer_tokens['access']}"},
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_history_requires_authentication(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/history")
        assert response.status_code == 401
