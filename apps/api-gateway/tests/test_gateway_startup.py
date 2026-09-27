"""Start-up of the gateway: seeding roles and demo users, and the application lifespan."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
import uvicorn
from api_gateway import __main__ as launcher
from api_gateway import db_init, main
from api_gateway.auth.password import verify_password
from api_gateway.config import settings
from api_gateway.models.user import Base, Role, User, UserRole
from api_gateway.realtime.hub import RealtimeHub
from gateway_fakes import (
    FakeAlarmClient,
    FakeHistorianClient,
    FakePhysicsClient,
    FakePLCClient,
)
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DEMO_PASSWORDS = {
    "demo_admin_password": "admin-demo-pass-1",
    "demo_engineer_password": "engineer-demo-pass-1",
    "demo_operator_password": "operator-demo-pass-1",
    "demo_viewer_password": "viewer-demo-pass-1",
}


@pytest_asyncio.fixture
async def seeded_database(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_init, "AsyncSessionLocal", factory)
    for field, value in DEMO_PASSWORDS.items():
        monkeypatch.setattr(settings, field, value)
    yield factory
    await engine.dispose()


async def roles_by_user(factory: async_sessionmaker[AsyncSession]) -> dict[str, str]:
    async with factory() as db:
        rows = await db.execute(
            select(User.username, Role.name)
            .join(UserRole, UserRole.user_id == User.id)
            .join(Role, Role.id == UserRole.role_id)
        )
        return {str(username): str(role) for username, role in rows}


class TestSeeding:
    async def test_every_role_and_demo_user_is_created(
        self, seeded_database: async_sessionmaker[AsyncSession]
    ) -> None:
        await db_init.seed_roles_and_demo_users()
        assert await roles_by_user(seeded_database) == {
            "admin": "admin",
            "engineer": "engineer",
            "operator": "operator",
            "viewer": "viewer",
        }
        async with seeded_database() as db:
            user = await db.scalar(select(User).where(User.username == "operator"))
        assert user is not None and user.is_active
        assert verify_password("operator-demo-pass-1", user.hashed_password)

    async def test_seeding_twice_changes_nothing(
        self, seeded_database: async_sessionmaker[AsyncSession]
    ) -> None:
        await db_init.seed_roles_and_demo_users()
        await db_init.seed_roles_and_demo_users()
        async with seeded_database() as db:
            counts = [
                await db.scalar(select(func.count()).select_from(table))
                for table in (Role, User, UserRole)
            ]
        assert counts == [4, 4, 4]

    async def test_an_existing_password_is_not_overwritten(
        self,
        seeded_database: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await db_init.seed_roles_and_demo_users()
        monkeypatch.setattr(settings, "demo_viewer_password", "changed-in-env-9")
        await db_init.seed_roles_and_demo_users()
        async with seeded_database() as db:
            user = await db.scalar(select(User).where(User.username == "viewer"))
        assert user is not None
        assert verify_password("viewer-demo-pass-1", user.hashed_password)

    async def test_a_user_without_a_password_is_skipped_with_a_warning(
        self,
        seeded_database: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(settings, "demo_admin_password", "")
        with caplog.at_level(logging.WARNING, logger="api_gateway.db_init"):
            await db_init.seed_roles_and_demo_users()
        assert "admin" not in await roles_by_user(seeded_database)
        assert "Demo user 'admin' not seeded" in caplog.text

    async def test_a_user_with_no_role_gets_its_demo_role_back(
        self,
        seeded_database: async_sessionmaker[AsyncSession],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await db_init.seed_roles_and_demo_users()
        async with seeded_database() as db:
            engineer = await db.scalar(select(User).where(User.username == "engineer"))
            assert engineer is not None
            assignment = await db.scalar(
                select(UserRole).where(UserRole.user_id == engineer.id)
            )
            await db.delete(assignment)
            await db.commit()
        with caplog.at_level(logging.WARNING, logger="api_gateway.db_init"):
            await db_init.seed_roles_and_demo_users()
        assert (await roles_by_user(seeded_database))["engineer"] == "engineer"
        assert "'engineer'" in caplog.text

    async def test_a_demoted_demo_user_keeps_the_role_an_admin_gave_it(
        self, seeded_database: async_sessionmaker[AsyncSession]
    ) -> None:
        await db_init.seed_roles_and_demo_users()
        async with seeded_database() as db:
            engineer = await db.scalar(select(User).where(User.username == "engineer"))
            viewer_role = await db.scalar(select(Role).where(Role.name == "viewer"))
            assert engineer is not None and viewer_role is not None
            await db.execute(delete(UserRole).where(UserRole.user_id == engineer.id))
            db.add(
                UserRole(user_id=engineer.id, role_id=viewer_role.id, granted_at_ms=1)
            )
            await db.commit()
        await db_init.seed_roles_and_demo_users()
        assert (await roles_by_user(seeded_database))["engineer"] == "viewer"


class TestLifespan:
    async def test_clients_and_sources_live_as_long_as_the_application(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made: dict[str, Any] = {}

        def keep(name: str, fake: Any) -> Any:
            def factory(config: object) -> Any:
                made[name] = (fake, config)
                return fake

            return factory

        closed: list[str] = []

        class Closing(FakePhysicsClient):
            async def close(self) -> None:
                closed.append("physics")

        class ClosingHistorian(FakeHistorianClient):
            def close(self) -> None:
                closed.append("historian")

        started = asyncio.Event()
        cancelled: list[str] = []

        async def source(name: str) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        async def telemetry(hub: RealtimeHub, physics: Any) -> None:
            await source("telemetry")

        async def plc_status(hub: RealtimeHub, plc: Any, interval_s: float) -> None:
            await source("plc")

        async def mqtt_events(hub: RealtimeHub, *args: Any) -> None:
            await source("mqtt")

        monkeypatch.setattr(main, "PhysicsGatewayClient", keep("physics", Closing()))
        monkeypatch.setattr(main, "PLCGatewayClient", keep("plc", FakePLCClient()))
        monkeypatch.setattr(
            main, "AlarmGatewayClient", keep("alarm", FakeAlarmClient())
        )
        monkeypatch.setattr(
            main, "HistorianQueryClient", keep("historian", ClosingHistorian())
        )
        monkeypatch.setattr(main, "run_telemetry", telemetry)
        monkeypatch.setattr(main, "run_plc_status", plc_status)
        monkeypatch.setattr(main, "run_mqtt_events", mqtt_events)
        monkeypatch.setattr(settings, "auto_init_db", False)

        app = main.create_app()
        async with main.lifespan(app):
            await asyncio.wait_for(started.wait(), timeout=5.0)
            assert isinstance(app.state.realtime_hub, RealtimeHub)
            assert app.state.physics_client is made["physics"][0]
            assert made["plc"][1].target == settings.plc_grpc_target
            assert made["historian"][1].bucket == settings.influx_bucket
        assert sorted(cancelled) == ["mqtt", "plc", "telemetry"]
        assert closed == ["physics", "historian"]

    async def test_auto_init_seeds_before_serving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded: list[bool] = []

        async def seed() -> None:
            seeded.append(True)

        async def idle(*_: Any) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(
            main, "PhysicsGatewayClient", lambda config: FakePhysicsClient()
        )
        monkeypatch.setattr(main, "PLCGatewayClient", lambda config: FakePLCClient())
        monkeypatch.setattr(
            main, "AlarmGatewayClient", lambda config: FakeAlarmClient()
        )
        monkeypatch.setattr(
            main, "HistorianQueryClient", lambda config: FakeHistorianClient()
        )
        for name in ("run_telemetry", "run_plc_status", "run_mqtt_events"):
            monkeypatch.setattr(main, name, idle)
        monkeypatch.setattr(main, "seed_roles_and_demo_users", seed)
        monkeypatch.setattr(settings, "auto_init_db", True)
        async with main.lifespan(main.create_app()):
            assert seeded == [True]

    async def test_missing_signing_keys_stop_start_up_before_anything_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made: list[str] = []

        def refuse(config: object) -> Any:
            made.append("client")
            return FakePhysicsClient()

        monkeypatch.setattr(main, "PhysicsGatewayClient", refuse)
        monkeypatch.setattr(settings, "auto_init_db", False)
        monkeypatch.setattr(settings, "jwt_private_key", "")
        with pytest.raises(RuntimeError, match="jwt_private_key"):
            async with main.lifespan(main.create_app()):
                pass
        assert made == []


class TestLauncher:
    @pytest.fixture
    def launched(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def run(app: str, **options: Any) -> None:
            calls.append({"app": app, **options})

        monkeypatch.setattr(launcher.uvicorn, "run", run)
        monkeypatch.setattr(launcher, "configure_logging", lambda service: None)
        return calls

    def test_the_launcher_passes_host_and_port_to_uvicorn(
        self, launched: list[dict[str, Any]]
    ) -> None:
        launcher.main(["--host", "0.0.0.0", "--port", "9000"])
        (options,) = launched
        assert options["app"] == "api_gateway.main:app"
        assert (options["host"], options["port"]) == ("0.0.0.0", 9000)
        assert options["forwarded_allow_ips"] == "127.0.0.1"
        assert options["proxy_headers"] is True

    def test_websocket_frames_and_queues_are_bounded(
        self, launched: list[dict[str, Any]]
    ) -> None:
        launcher.main([])
        (options,) = launched
        assert options["ws_max_size"] <= 64 * 1024
        assert options["ws_max_queue"] <= 8

    @pytest.mark.parametrize(
        ("platform", "loop"),
        [("win32", "asyncio:SelectorEventLoop"), ("linux", "auto")],
    )
    def test_windows_gets_the_selector_loop_mqtt_needs(
        self,
        launched: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
        loop: str,
    ) -> None:
        monkeypatch.setattr(launcher.sys, "platform", platform)
        launcher.main([])
        assert launched[0]["loop"] == loop

    def test_uvicorn_resolves_the_loop_name_to_a_selector_loop(self) -> None:
        config = uvicorn.Config("api_gateway.main:app", loop=launcher.WINDOWS_LOOP)
        assert config.get_loop_factory() is asyncio.SelectorEventLoop
