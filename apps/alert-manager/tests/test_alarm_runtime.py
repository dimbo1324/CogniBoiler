"""AlarmService over gRPC, the change publisher, the MQTT intake and the entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from aiomqtt import MqttError
from alarm_factories import Recorder, condition, condition_payload, only_alarm
from alert_manager import __main__ as entry
from alert_manager import db, publisher, subscriber
from alert_manager.grpc_server import AlarmServicer, start_server
from alert_manager.lifecycle import AlarmState
from alert_manager.models import Base
from alert_manager.processor import AlarmProcessor
from alert_manager.publisher import AlarmChangePublisher
from alert_manager.subscriber import AlertSubscriber
from alert_manager.views import AlarmView, TransitionView
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest_asyncio.fixture
async def stub(
    processor: AlarmProcessor,
) -> AsyncIterator[tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]]]:
    subscribed = {"value": True}
    server = grpc.aio.server()
    pb2_grpc.add_AlarmServiceServicer_to_server(
        AlarmServicer(processor, is_subscribed=lambda: subscribed["value"]), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield pb2_grpc.AlarmServiceStub(channel), subscribed
    finally:
        await channel.close()
        await server.stop(grace=None)


class TestAlarmService:
    async def test_health_follows_the_broker_and_the_database(
        self,
        stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]],
        processor: AlarmProcessor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, subscribed = stub
        health = await client.Health(pb2.Empty())
        assert (health.service, health.status) == ("alert-manager", "running")
        subscribed["value"] = False
        assert (await client.Health(pb2.Empty())).status == "degraded"

        async def unreachable() -> None:
            raise OperationalError("SELECT 1", {}, Exception("refused"))

        subscribed["value"] = True
        monkeypatch.setattr(processor, "ping", unreachable)
        assert (await client.Health(pb2.Empty())).status == "degraded"

    async def test_lists_details_and_acknowledgements(
        self,
        stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]],
        processor: AlarmProcessor,
    ) -> None:
        client, _ = stub
        await processor.handle_condition(condition("water_level_m"))
        await processor.handle_condition(condition("stack_temp_k", severity="warning"))
        listed = await client.ListAlarms(pb2.ListAlarmsRequest(open_only=True))
        assert listed.total == 2
        assert listed.alarms[0].severity == "critical"
        assert listed.alarms[0].state == pb2.AlarmState.ALARM_ACTIVE_UNACK
        alarm_id = listed.alarms[0].alarm_id

        acknowledged = await client.AcknowledgeAlarm(
            pb2.AcknowledgeAlarmRequest(
                alarm_id=alarm_id, operator_id="operator1", comment="seen"
            )
        )
        assert acknowledged.accepted is True
        assert acknowledged.alarms[0].acknowledged_by == "operator1"
        assert acknowledged.alarms[0].state == pb2.AlarmState.ALARM_ACTIVE_ACK

        again = await client.AcknowledgeAlarm(
            pb2.AcknowledgeAlarmRequest(alarm_id=alarm_id, operator_id="operator2")
        )
        assert (again.accepted, again.reason) == (
            False,
            "alarm is already acknowledged",
        )

        detail = await client.GetAlarm(pb2.AlarmRef(alarm_id=alarm_id))
        assert [t.to_state for t in detail.transitions] == [
            pb2.AlarmState.ALARM_ACTIVE_UNACK,
            pb2.AlarmState.ALARM_ACTIVE_ACK,
        ]
        assert (
            detail.transitions[0].from_state == pb2.AlarmState.ALARM_STATE_UNSPECIFIED
        )
        assert detail.transitions[1].comment == "seen"

        rest = await client.AcknowledgeAll(
            pb2.AcknowledgeAllRequest(operator_id="operator1", severity="warning")
        )
        assert [alarm.parameter for alarm in rest.alarms] == ["stack_temp_k"]

    async def test_an_unknown_alarm_is_not_found(
        self, stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]]
    ) -> None:
        client, _ = stub
        for call in (
            client.GetAlarm(pb2.AlarmRef(alarm_id=404)),
            client.AcknowledgeAlarm(pb2.AcknowledgeAlarmRequest(alarm_id=404)),
        ):
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call
            assert failed.value.code() == grpc.StatusCode.NOT_FOUND

    async def test_an_unknown_severity_is_an_invalid_argument(
        self, stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]]
    ) -> None:
        client, _ = stub
        for call in (
            client.ListAlarms(pb2.ListAlarmsRequest(severity="info")),
            client.AcknowledgeAll(pb2.AcknowledgeAllRequest(severity="info")),
        ):
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call
            assert failed.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_the_server_starts_on_its_port(
        self, processor: AlarmProcessor
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = await start_server(
            AlarmServicer(processor, is_subscribed=lambda: True), port
        )
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                health = await pb2_grpc.AlarmServiceStub(channel).Health(pb2.Empty())
        finally:
            await server.stop(grace=None)
        assert health.status == "running"


def change(alarm_id: int) -> tuple[AlarmView, TransitionView]:
    view = AlarmView(
        id=alarm_id,
        key=f"k{alarm_id}",
        source_service="plc-controller",
        parameter="water_level_m",
        severity="critical",
        direction="low",
        unit="m",
        state=AlarmState.ACTIVE_UNACK,
        message="m",
        action="trip",
        topic="alerts/critical",
        value=1.0,
        threshold=2.0,
        raised_at_ms=1,
        cleared_at_ms=None,
        acknowledged_at_ms=None,
        acknowledged_by=None,
        ack_comment=None,
        occurrence_count=1,
        updated_at_ms=1,
    )
    transition = TransitionView(
        id=alarm_id,
        alarm_id=alarm_id,
        from_state=None,
        to_state=AlarmState.ACTIVE_UNACK,
        at_ms=alarm_id,
        actor="plc-controller",
        comment=None,
        value=1.0,
    )
    return view, transition


class FakeBroker:
    """aiomqtt.Client stand-in shared by the publisher and subscriber tests."""

    up = True
    connections: list[dict[str, Any]] = []
    published: list[tuple[str, bytes, int]] = []
    subscriptions: list[tuple[str, int]] = []
    deliveries: list[SimpleNamespace] = []

    def __init__(self, **options: Any) -> None:
        FakeBroker.connections.append(options)

    async def __aenter__(self) -> FakeBroker:
        if not FakeBroker.up:
            raise MqttError("broker unreachable")
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def publish(self, topic: str, payload: bytes, qos: int) -> None:
        FakeBroker.published.append((topic, payload, qos))

    async def subscribe(self, topic: str, qos: int) -> None:
        FakeBroker.subscriptions.append((topic, qos))

    @property
    def messages(self) -> AsyncIterator[SimpleNamespace]:
        async def deliver() -> AsyncIterator[SimpleNamespace]:
            for message in FakeBroker.deliveries:
                yield message
            raise MqttError("connection lost")

        return deliver()


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> type[FakeBroker]:
    FakeBroker.up = True
    FakeBroker.connections = []
    FakeBroker.published = []
    FakeBroker.subscriptions = []
    FakeBroker.deliveries = []
    monkeypatch.setattr(publisher, "Client", FakeBroker)
    monkeypatch.setattr(subscriber, "Client", FakeBroker)
    monkeypatch.setattr(publisher, "RECONNECT_DELAY_S", 0.001)
    monkeypatch.setattr(subscriber, "RECONNECT_DELAY_S", 0.001)
    monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
    return FakeBroker


async def until(predicate: Any) -> None:
    async with asyncio.timeout(5.0):
        while not predicate():
            await asyncio.sleep(0.001)


class TestPublisher:
    async def test_changes_go_out_in_order_on_the_changes_topic(
        self, broker: type[FakeBroker]
    ) -> None:
        sender = AlarmChangePublisher(
            "broker", 1883, username="alert-manager", password="pw"
        )
        for alarm_id in (1, 2, 3):
            sender.alarm_changed(*change(alarm_id))
        sender.start()
        sender.start()
        await until(lambda: len(broker.published) == 3)
        await sender.aclose()
        ids = [json.loads(payload)["alarm"]["id"] for _, payload, _ in broker.published]
        assert ids == [1, 2, 3]
        assert {(topic, qos) for topic, _, qos in broker.published} == {
            ("alarms/changes", 1)
        }
        assert broker.connections[0]["username"] == "alert-manager"
        assert len(broker.connections) == 1

    async def test_changes_wait_for_the_broker(
        self, broker: type[FakeBroker], caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.up = False
        sender = AlarmChangePublisher("broker", 1883)
        with caplog.at_level(logging.INFO, logger="alert_manager.publisher"):
            sender.start()
            sender.alarm_changed(*change(1))
            await until(lambda: len(broker.connections) >= 3)
            assert broker.published == []
            broker.up = True
            await until(lambda: len(broker.published) == 1)
        await sender.aclose()
        assert caplog.text.count("lost MQTT") == 1
        assert "reconnected" in caplog.text

    async def test_a_full_queue_drops_the_oldest(
        self,
        broker: type[FakeBroker],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(publisher, "QUEUE_LIMIT", 2)
        sender = AlarmChangePublisher("broker", 1883)
        with caplog.at_level(logging.ERROR, logger="alert_manager.publisher"):
            for alarm_id in (1, 2, 3, 4):
                sender.alarm_changed(*change(alarm_id))
        sender.start()
        await until(lambda: len(broker.published) == 2)
        await sender.aclose()
        ids = [json.loads(payload)["alarm"]["id"] for _, payload, _ in broker.published]
        assert ids == [3, 4]
        assert "2 changes dropped" not in caplog.text
        assert "1 changes dropped" in caplog.text

    async def test_closing_an_unstarted_publisher_is_harmless(self) -> None:
        await AlarmChangePublisher("broker", 1883).aclose()


class RecordingHandler:
    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.conditions: list[Any] = []
        self.snapshots: list[Any] = []
        self._failures = failures
        self._error = error

    async def handle_condition(self, report: Any) -> None:
        if self._failures:
            self._failures -= 1
            raise self._error or OperationalError("INSERT", {}, Exception("locked"))
        self.conditions.append(report)

    async def handle_snapshot(self, report: Any) -> None:
        self.snapshots.append(report)


def delivery(topic: str, payload: object) -> SimpleNamespace:
    return SimpleNamespace(topic=topic, payload=payload)


class TestSubscriber:
    async def test_conditions_and_snapshots_reach_the_processor(
        self, broker: type[FakeBroker]
    ) -> None:
        handler = RecordingHandler()
        broker.deliveries = [
            delivery("alerts/critical", condition_payload()),
            delivery(
                "alerts/snapshot",
                json.dumps(
                    {
                        "source_service": "plc-controller",
                        "active_keys": [],
                        "timestamp_ms": 1,
                    }
                ).encode(),
            ),
            delivery("alerts/warning", b"not json"),
            delivery("alerts/warning", "a text payload"),
        ]
        intake = AlertSubscriber(
            "broker", 1883, handler, username="alert-manager", password="pw"
        )
        task = asyncio.create_task(intake.run())
        try:
            await until(lambda: intake.stats["received"] >= 4)
        finally:
            task.cancel()
        assert len(handler.conditions) == 1
        assert len(handler.snapshots) == 1
        assert intake.stats == {
            "received": 4,
            "processed": 2,
            "skipped": 2,
            "failed": 0,
        }
        assert broker.subscriptions[0] == ("alerts/#", 1)
        assert broker.connections[0]["clean_session"] is False
        assert broker.connections[0]["identifier"] == "alert-manager"

    async def test_the_connected_flag_follows_the_session(
        self, broker: type[FakeBroker]
    ) -> None:
        intake = AlertSubscriber("broker", 1883, RecordingHandler())
        broker.up = False
        task = asyncio.create_task(intake.run())
        try:
            await until(lambda: len(broker.connections) >= 2)
            assert intake.connected is False
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_a_database_hiccup_is_retried(self) -> None:
        handler = RecordingHandler(failures=2)
        intake = AlertSubscriber(handler=handler)
        await intake._handle_message("alerts/critical", condition_payload())
        assert len(handler.conditions) == 1
        assert intake.stats["processed"] == 1

    async def test_a_message_is_given_up_after_the_retries(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
        handler = RecordingHandler(failures=5)
        intake = AlertSubscriber(handler=handler)
        with caplog.at_level(logging.WARNING, logger="alert_manager.subscriber"):
            await intake._handle_message("alerts/critical", condition_payload())
        assert intake.stats["failed"] == 1
        assert caplog.text.count("attempt") == 2
        assert "could not be processed" in caplog.text

    async def test_an_unexpected_error_is_not_retried(self) -> None:
        handler = RecordingHandler(failures=1, error=RuntimeError("bug"))
        intake = AlertSubscriber(handler=handler)
        await intake._handle_message("alerts/critical", condition_payload())
        assert (intake.stats["failed"], handler.conditions) == (1, [])

    async def test_without_a_handler_messages_are_skipped(self) -> None:
        intake = AlertSubscriber()
        await intake._handle_message("alerts/critical", condition_payload())
        assert intake.stats["skipped"] == 1

    async def test_end_to_end_into_the_processor(
        self, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        intake = AlertSubscriber(handler=processor)
        await intake._handle_message("alerts/critical", condition_payload())
        alarm = await only_alarm(processor)
        assert alarm.key == "plc-controller:water_level_m:low:critical"
        assert recorder.states == [(None, "ACTIVE_UNACK")]


class TestDatabaseAccess:
    def test_the_url_prefers_the_service_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ALERT_MANAGER_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert db.database_url() == db.DEFAULT_DATABASE_URL
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://shared/db")
        assert db.database_url() == "postgresql+asyncpg://shared/db"
        monkeypatch.setenv(
            "ALERT_MANAGER_DATABASE_URL", "postgresql+asyncpg://alarms/db"
        )
        assert db.database_url() == "postgresql+asyncpg://alarms/db"

    async def test_missing_tables_are_reported_until_migrated(
        self, tmp_path: Path
    ) -> None:
        engine = db.create_engine(
            f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}"
        )
        assert await db.missing_tables(engine) == ["alarm_events", "alarm_transitions"]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        assert await db.missing_tables(engine) == []
        factory = db.session_factory(engine)
        async with factory() as session:
            assert isinstance(session, AsyncSession)
        await engine.dispose()


class TestEntryPoint:
    def _args(self, liveness: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            mqtt_host="broker",
            mqtt_port=1883,
            grpc_port=0,
            liveness_file=liveness,
            metrics_port=0,
            metrics_host="127.0.0.1",
        )

    async def test_it_refuses_to_start_before_the_migrations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        url = f"sqlite+aiosqlite:///{(tmp_path / 'empty.db').as_posix()}"
        monkeypatch.setattr(entry, "create_engine", lambda: create_async_engine(url))
        with caplog.at_level(logging.ERROR, logger="alert_manager"):
            assert await entry.main(self._args()) == 1
        assert "Alarm tables missing: alarm_events, alarm_transitions" in caplog.text

    async def test_it_wires_intake_processor_publisher_and_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"sqlite+aiosqlite:///{(tmp_path / 'alarms.db').as_posix()}"
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
        monkeypatch.setattr(entry, "create_engine", lambda: create_async_engine(url))
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        monkeypatch.setenv("MQTT_PASSWORD", "broker-pass")
        events: list[str] = []

        class Publisher:
            def __init__(self, host: str, port: int, **options: Any) -> None:
                events.append(f"publisher {options['username']} {options['password']}")

            def start(self) -> None:
                events.append("publisher started")

            async def aclose(self) -> None:
                events.append("publisher closed")

        class Intake:
            connected = True

            def __init__(self, host: str, port: int, handler: Any, **_: Any) -> None:
                assert isinstance(handler, AlarmProcessor)

            async def run(self) -> None:
                events.append("intake ran")

        class Server:
            async def stop(self, grace: float) -> None:
                events.append(f"server stopped {grace}")

        async def start(servicer: AlarmServicer, port: int) -> Server:
            events.append(f"server started {port}")
            return Server()

        monkeypatch.setattr(entry, "AlarmChangePublisher", Publisher)
        monkeypatch.setattr(entry, "AlertSubscriber", Intake)
        monkeypatch.setattr(entry, "start_server", start)
        assert await entry.main(self._args()) == 0
        assert events[:3] == [
            "publisher alert-manager broker-pass",
            "publisher started",
            "server started 0",
        ]
        assert "intake ran" in events
        assert events[-2:] == ["server stopped 5", "publisher closed"]

    def test_the_command_line_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["alert_manager"])
        args = entry.parse_args()
        assert (args.grpc_port, args.metrics_port, args.liveness_file) == (
            50053,
            9104,
            None,
        )


async def test_a_processor_over_a_real_file_database(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    recorder = Recorder()
    found = AlarmProcessor(sessions, recorder, clear_hold_s=0.0)
    await found.handle_condition(condition())
    await found.close()
    assert recorder.states == [(None, "ACTIVE_UNACK")]
