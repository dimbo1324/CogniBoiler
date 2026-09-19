"""WebSocket /ws end to end, through Starlette's test client.

The test client runs the application on an event loop of its own, so the database is a
SQLite file opened per session instead of the in-memory database of the HTTP tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from api_gateway.auth.identity import load_account
from api_gateway.auth.jwt_handler import issue_access_token
from api_gateway.auth.sessions import ClientInfo, open_session, revoke_user_sessions
from api_gateway.config import settings
from api_gateway.dependencies import get_db
from api_gateway.main import create_app
from api_gateway.models.user import AuditLog, Base
from api_gateway.realtime.hub import Channel, RealtimeHub
from api_gateway.routers import websocket as ws_router
from fastapi import FastAPI
from gateway_fakes import seed_accounts
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@dataclass
class Session:
    user_id: int
    session_id: str
    access: str


@dataclass
class Live:
    app: FastAPI
    hub: RealtimeHub
    client: TestClient
    database_url: str
    sessions: dict[str, Session]

    def run(self, work: Any) -> Any:
        """Run a coroutine function against the database on a loop of its own."""

        async def scoped() -> Any:
            engine = create_async_engine(self.database_url, poolclass=NullPool)
            try:
                async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                    return await work(db)
            finally:
                await engine.dispose()

        return asyncio.run(scoped())

    def audit(self) -> list[AuditLog]:
        async def rows(db: AsyncSession) -> list[AuditLog]:
            return list((await db.execute(select(AuditLog))).scalars().all())

        result: list[AuditLog] = self.run(rows)
        return result


@pytest.fixture
def live(tmp_path: Path) -> Iterator[Live]:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'gateway.db').as_posix()}"

    async def prepare() -> dict[str, Session]:
        engine = create_async_engine(database_url, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        opened = {}
        async with factory() as db:
            await seed_accounts(db)
            for username in ("viewer1", "operator1"):
                account = await load_account(db, username=username)
                assert account is not None
                tokens = await open_session(
                    db, account, ClientInfo(ip="127.0.0.1", user_agent="pytest")
                )
                opened[username] = Session(
                    account.id, tokens.refresh.session_id, tokens.access.token
                )
        await engine.dispose()
        return opened

    sessions = asyncio.run(prepare())
    app = create_app()
    hub = RealtimeHub(queue_size=8, max_rate_hz=10.0)
    app.state.realtime_hub = hub

    async def per_session_database() -> AsyncGenerator[AsyncSession]:
        engine = create_async_engine(database_url, poolclass=NullPool)
        try:
            async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                yield db
        finally:
            await engine.dispose()

    app.dependency_overrides[get_db] = per_session_database
    yield Live(app, hub, TestClient(app), database_url, sessions)
    app.dependency_overrides.clear()


def authenticate(connection: Any, token: str) -> dict[str, Any]:
    connection.send_json({"type": "auth", "access_token": token})
    welcome: dict[str, Any] = connection.receive_json()
    return welcome


def leave(live: Live, connection: Any) -> None:
    """
    Close from the client and wait until the server has let the subscriber go.

    Leaving the test client's block while the handler still runs cancels the handler
    from outside, which no real server does; a client close is what a browser sends.
    """
    connection.close(1000)
    deadline = time.monotonic() + 5.0
    while live.hub.subscriber_count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert live.hub.subscriber_count == 0


def close_code(connection: Any) -> tuple[int, str]:
    """Read until the server closes; the code and reason it closed with."""
    while True:
        try:
            connection.receive_json()
        except WebSocketDisconnect as closed:
            return closed.code, closed.reason


class TestAuthentication:
    def test_a_valid_first_frame_is_welcomed(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            welcome = authenticate(connection, live.sessions["viewer1"].access)
            leave(live, connection)
        assert welcome["type"] == "welcome"
        assert (welcome["user"], welcome["role"]) == ("viewer1", "viewer")
        assert welcome["channels"] == ["telemetry", "plc", "alarms"]
        assert welcome["max_rate_hz"] == 10.0
        assert welcome["token_expires_at_ms"] > int(time.time() * 1000)

    def test_a_first_frame_that_is_not_auth_is_refused_and_audited(
        self, live: Live
    ) -> None:
        with live.client.websocket_connect("/ws") as connection:
            connection.send_json({"type": "subscribe", "channels": ["plc"]})
            code, reason = close_code(connection)
        assert (code, reason) == (4401, "auth.token_missing")
        (row,) = live.audit()
        assert (row.method, row.endpoint, row.response_status) == ("WS", "/ws", 401)
        assert row.outcome == "refused: auth.token_missing"

    def test_an_invalid_token_is_refused(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            connection.send_json({"type": "auth", "access_token": "not.a.token"})
            assert close_code(connection) == (4401, "auth.token_invalid")

    def test_a_frame_that_is_not_json_is_refused(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            connection.send_text("hello")
            assert close_code(connection) == (4401, "ws.bad_request")

    def test_an_oversized_frame_is_refused(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            connection.send_text('{"type": "auth", "pad": "' + "x" * 9000 + '"}')
            assert close_code(connection) == (4401, "ws.bad_request")

    def test_no_first_frame_in_time_is_refused(
        self, live: Live, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "ws_auth_timeout_s", 0.05)
        with live.client.websocket_connect("/ws") as connection:
            assert close_code(connection) == (4401, "ws.auth_timeout")

    def test_without_a_hub_the_channel_is_unavailable(self, live: Live) -> None:
        del live.app.state.realtime_hub
        with live.client.websocket_connect("/ws") as connection:
            assert close_code(connection) == (1013, "realtime unavailable")


class TestSession:
    def test_subscriptions_start_from_the_latest_snapshot(self, live: Live) -> None:
        live.hub.publish(Channel.PLC, "status", {"mode": "auto"})
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json(
                {"type": "subscribe", "channels": ["plc", "alarms"], "max_rate_hz": 2}
            )
            replies = [connection.receive_json(), connection.receive_json()]
            connection.portal.call(
                live.hub.publish, Channel.ALARMS, "change", {"alarm_id": 7}
            )
            change = connection.receive_json()
            leave(live, connection)
        kinds = {reply["type"] for reply in replies}
        assert kinds == {"data", "subscribed"}
        subscribed = next(reply for reply in replies if reply["type"] == "subscribed")
        assert subscribed["channels"] == ["alarms", "plc"]
        snapshot = next(reply for reply in replies if reply["type"] == "data")
        assert snapshot["data"] == {"mode": "auto"}
        assert (change["channel"], change["data"]) == ("alarms", {"alarm_id": 7})

    def test_unsubscribing_stops_the_channel(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json({"type": "subscribe", "channels": ["alarms"]})
            connection.receive_json()
            connection.send_json({"type": "unsubscribe", "channels": ["alarms"]})
            reply = connection.receive_json()
            leave(live, connection)
        assert reply == {"type": "subscribed", "channels": []}

    def test_ping_and_unknown_messages(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json({"type": "ping"})
            pong = connection.receive_json()
            connection.send_json({"type": "dance"})
            error = connection.receive_json()
            leave(live, connection)
        assert pong["type"] == "pong"
        assert error["code"] == "ws.unknown_message"

    @pytest.mark.parametrize("channels", [["weather"], [], "plc"])
    def test_a_bad_channel_list_closes_with_4400(
        self, live: Live, channels: object
    ) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json({"type": "subscribe", "channels": channels})
            code, _ = close_code(connection)
        assert code == 4400

    def test_a_frame_that_is_not_an_object_closes_with_4400(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json([1, 2, 3])
            assert close_code(connection) == (4400, "frames must be JSON objects")

    def test_a_fresh_token_of_the_same_user_renews_the_session(
        self, live: Live
    ) -> None:
        viewer = live.sessions["viewer1"]
        fresh = issue_access_token(viewer.user_id, "viewer", viewer.session_id)
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, viewer.access)
            connection.send_json({"type": "auth", "access_token": fresh.token})
            renewed = connection.receive_json()
            leave(live, connection)
        # The token carries its expiry in whole seconds (JWT exp).
        assert renewed == {
            "type": "renewed",
            "token_expires_at_ms": fresh.expires_at_ms // 1000 * 1000,
        }

    def test_a_token_of_another_user_closes_the_connection(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json(
                {"type": "auth", "access_token": live.sessions["operator1"].access}
            )
            assert close_code(connection) == (4401, "token belongs to another user")

    def test_an_invalid_renewal_closes_the_connection(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json({"type": "auth", "access_token": ""})
            assert close_code(connection) == (4401, "auth.token_missing")


class TestGuard:
    def test_an_expired_token_closes_the_connection(self, live: Live) -> None:
        viewer = live.sessions["viewer1"]
        short = issue_access_token(
            viewer.user_id,
            "viewer",
            viewer.session_id,
            not_after_ms=int(time.time() * 1000) + 1500,
        )
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, short.token)
            code, reason = close_code(connection)
        assert (code, reason) == (4401, "token expired")

    def test_a_closed_session_closes_the_connection(
        self, live: Live, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ws_router, "REVALIDATE_INTERVAL_S", 0.05)
        viewer = live.sessions["viewer1"]
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, viewer.access)

            async def sign_out(db: AsyncSession) -> None:
                await revoke_user_sessions(db, viewer.user_id, "admin")
                await db.commit()

            live.run(sign_out)
            code, _ = close_code(connection)
        assert code == 4401

    def test_a_client_that_cannot_keep_up_is_told_to_reload(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            connection.send_json({"type": "subscribe", "channels": ["plc"]})
            connection.receive_json()

            def flood() -> None:
                for number in range(20):
                    live.hub.publish(Channel.PLC, "event", {"number": number})

            connection.portal.call(flood)
            closing: dict[str, Any] = {}
            while not closing:
                try:
                    frame = connection.receive_json()
                except WebSocketDisconnect as closed:
                    assert closed.code == 1013
                    break
                if frame["type"] == "closing":
                    closing = frame
        assert closing == {
            "type": "closing",
            "code": 1013,
            "reason": "client too slow; reload state",
        }

    def test_the_subscriber_leaves_the_hub_on_disconnect(self, live: Live) -> None:
        with live.client.websocket_connect("/ws") as connection:
            authenticate(connection, live.sessions["viewer1"].access)
            assert live.hub.subscriber_count == 1
            leave(live, connection)
