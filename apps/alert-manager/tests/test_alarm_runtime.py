"""AlarmService over gRPC, the change publisher, the MQTT intake and the entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
from alert_manager import db, publisher, service, subscriber
from alert_manager.grpc_server import AlarmServicer, start_server
from alert_manager.lifecycle import AlarmState
from alert_manager.models import Base
from alert_manager.processor import AlarmProcessor
from alert_manager.publisher import AlarmChangePublisher
from alert_manager.queries import AlarmQueries
from alert_manager.subscriber import AlertSubscriber
from alert_manager.views import AlarmView, TransitionView
from prometheus_client import REGISTRY
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    InterfaceError,
    OperationalError,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest_asyncio.fixture
async def stub(
    queries: AlarmQueries,
    processor: AlarmProcessor,
) -> AsyncIterator[tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]]]:
    subscribed = {"value": True}
    server = grpc.aio.server()
    pb2_grpc.add_AlarmServiceServicer_to_server(
        AlarmServicer(processor, queries, is_subscribed=lambda: subscribed["value"]),
        server,
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
        queries: AlarmQueries,
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
        monkeypatch.setattr(queries, "ping", unreachable)
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
        for alarm_id in (404, 2**31, 2**40):
            for call in (
                client.GetAlarm(pb2.AlarmRef(alarm_id=alarm_id)),
                client.AcknowledgeAlarm(
                    pb2.AcknowledgeAlarmRequest(
                        alarm_id=alarm_id, operator_id="operator1"
                    )
                ),
            ):
                with pytest.raises(grpc.aio.AioRpcError) as failed:
                    await call
                assert failed.value.code() == grpc.StatusCode.NOT_FOUND

    async def test_an_acknowledgement_without_an_operator_is_an_invalid_argument(
        self,
        queries: AlarmQueries,
        stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]],
        processor: AlarmProcessor,
    ) -> None:
        client, _ = stub
        await processor.handle_condition(condition())
        alarm = await only_alarm(queries)
        for call in (
            client.AcknowledgeAlarm(
                pb2.AcknowledgeAlarmRequest(alarm_id=alarm.id, operator_id="  ")
            ),
            client.AcknowledgeAll(pb2.AcknowledgeAllRequest()),
        ):
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call
            assert failed.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert (await only_alarm(queries)).state is AlarmState.ACTIVE_UNACK

    async def test_an_acknowledgement_logs_the_calling_peer(
        self,
        queries: AlarmQueries,
        stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]],
        processor: AlarmProcessor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client, _ = stub
        await processor.handle_condition(condition())
        alarm = await only_alarm(queries)
        with caplog.at_level(logging.INFO, logger="alert_manager.grpc_server"):
            await client.AcknowledgeAlarm(
                pb2.AcknowledgeAlarmRequest(alarm_id=alarm.id, operator_id="operator1")
            )
            await client.AcknowledgeAll(
                pb2.AcknowledgeAllRequest(operator_id="operator1")
            )
        lines = [
            r.getMessage() for r in caplog.records if r.name.endswith("grpc_server")
        ]
        assert len(lines) == 2
        assert all("operator1" in line and "127.0.0.1" in line for line in lines)

    async def test_an_unknown_severity_is_an_invalid_argument(
        self, stub: tuple[pb2_grpc.AlarmServiceStub, dict[str, bool]]
    ) -> None:
        client, _ = stub
        for call in (
            client.ListAlarms(pb2.ListAlarmsRequest(severity="info")),
            client.AcknowledgeAll(
                pb2.AcknowledgeAllRequest(operator_id="operator1", severity="info")
            ),
        ):
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call
            assert failed.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test_the_server_starts_on_the_port_it_bound(
        self, processor: AlarmProcessor, queries: AlarmQueries
    ) -> None:
        # Port 0 and the bound port handed back: probing a free port and binding it
        # later races any other process for it.
        server, port = await start_server(
            AlarmServicer(processor, queries, is_subscribed=lambda: True), 0
        )
        try:
            assert port > 0
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
    fail_at: int | None = None
    publish_calls = 0
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

    async def publish(
        self, topic: str, payload: bytes, qos: int, retain: bool = False
    ) -> None:
        assert retain is False
        FakeBroker.publish_calls += 1
        if FakeBroker.publish_calls == FakeBroker.fail_at:
            raise MqttError("connection lost while publishing")
        FakeBroker.published.append((topic, payload, qos))

    async def subscribe(self, topic: str, qos: int) -> None:
        FakeBroker.subscriptions.append((topic, qos))

    async def unsubscribe(self, topic: str) -> None:
        FakeBroker.subscriptions.append((topic, -1))

    @property
    def messages(self) -> AsyncIterator[SimpleNamespace]:
        pending, FakeBroker.deliveries = FakeBroker.deliveries, []

        async def deliver() -> AsyncIterator[SimpleNamespace]:
            for message in pending:
                yield message
            raise MqttError("connection lost")

        return deliver()


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> type[FakeBroker]:
    FakeBroker.up = True
    FakeBroker.fail_at = None
    FakeBroker.publish_calls = 0
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
        # A broker that is away for three attempts costs one warning, not three.
        assert caplog.text.count("MQTT error") == 1
        assert "MQTT connection is back" in caplog.text

    async def test_a_full_queue_drops_the_oldest(
        self,
        broker: type[FakeBroker],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(publisher, "QUEUE_LIMIT", 2)
        dropped_before = REGISTRY.get_sample_value("alarm_changes_dropped_total")
        sender = AlarmChangePublisher("broker", 1883)
        with caplog.at_level(logging.ERROR, logger="alert_manager.publisher"):
            for alarm_id in (1, 2, 3, 4):
                sender.alarm_changed(*change(alarm_id))
        sender.start()
        await until(lambda: len(broker.published) == 2)
        await sender.aclose()
        ids = [json.loads(payload)["alarm"]["id"] for _, payload, _ in broker.published]
        assert ids == [3, 4]
        assert sender.dropped == 2
        assert REGISTRY.get_sample_value("alarm_changes_dropped_total") == (
            (dropped_before or 0.0) + 2
        )
        # The first drop of an outage is an error; the next ones up to the 100th are not.
        assert "2 messages dropped" not in caplog.text
        assert "1 messages dropped" in caplog.text

    async def test_closing_an_unstarted_publisher_is_harmless(self) -> None:
        await AlarmChangePublisher("broker", 1883).aclose()

    async def test_a_change_made_just_before_closing_is_still_published(
        self, broker: type[FakeBroker]
    ) -> None:
        # Committed to PostgreSQL already: dropping it would leave a permanent gap
        # in the historian's alarm history.
        sender = AlarmChangePublisher("broker", 1883)
        sender.start()
        await until(lambda: sender.connected)
        sender.alarm_changed(*change(1))
        sender.alarm_changed(*change(2))
        await sender.aclose()
        ids = [json.loads(payload)["alarm"]["id"] for _, payload, _ in broker.published]
        assert ids == [1, 2]

    async def test_closing_without_a_broker_says_what_is_lost(
        self, broker: type[FakeBroker], caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.up = False
        sender = AlarmChangePublisher("broker", 1883)
        sender.start()
        sender.alarm_changed(*change(1))
        await until(lambda: len(broker.connections) >= 1)
        with caplog.at_level(logging.WARNING, logger="alert_manager.publisher"):
            await sender.aclose()
        assert "1 alarm changes unpublished" in caplog.text
        assert broker.published == []

    async def test_a_publish_failing_mid_drain_is_resent_first(
        self, broker: type[FakeBroker]
    ) -> None:
        broker.fail_at = 2
        sender = AlarmChangePublisher("broker", 1883)
        for alarm_id in (1, 2, 3):
            sender.alarm_changed(*change(alarm_id))
        sender.start()
        await until(lambda: len(broker.published) == 3)
        await sender.aclose()
        ids = [json.loads(payload)["alarm"]["id"] for _, payload, _ in broker.published]
        assert ids == [1, 2, 3]
        assert len(broker.connections) == 2


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


def counts() -> dict[str, float]:
    def sample(name: str, labels: dict[str, str] | None = None) -> float:
        return REGISTRY.get_sample_value(name, labels) or 0.0

    return {
        "received": sum(
            sample("mqtt_messages_received_total", {"topic": topic})
            for topic in ("alerts/critical", "alerts/warning", "alerts/snapshot")
        ),
        "invalid": rejected("invalid"),
        "not_bytes": rejected("not_bytes"),
        "failed": sample("alarm_messages_failed_total"),
    }


def delta(before: dict[str, float]) -> dict[str, float]:
    return {name: value - before[name] for name, value in counts().items()}


def rejected(reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "alarm_messages_rejected_total", {"reason": reason}
    )
    return value or 0.0


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
        before = counts()
        task = asyncio.create_task(intake.run())
        try:
            await until(lambda: counts()["not_bytes"] > before["not_bytes"])
        finally:
            task.cancel()
        assert len(handler.conditions) == 1
        assert len(handler.snapshots) == 1
        assert delta(before) == {
            "received": 4,
            "invalid": 1,
            "not_bytes": 1,
            "failed": 0,
        }
        # The wildcard a persistent session may still hold from an older release
        # goes first; then exactly the three topics of the contract.
        assert broker.subscriptions[:4] == [
            ("alerts/#", -1),
            ("alerts/warning", 1),
            ("alerts/critical", 1),
            ("alerts/snapshot", 1),
        ]
        assert broker.connections[0]["clean_session"] is False
        assert broker.connections[0]["identifier"] == "alert-manager"
        assert (
            broker.connections[0]["max_queued_incoming_messages"]
            == subscriber.INCOMING_QUEUE_LIMIT
        )

    def test_a_subscriber_that_never_connected_is_not_connected(self) -> None:
        # What the container healthcheck reads before the first session opens.
        assert AlertSubscriber().connected is False
        assert AlertSubscriber().healthy is False

    async def test_a_message_stuck_in_processing_makes_it_unhealthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The broker session can stay up while a hung database holds the intake.
        release = asyncio.Event()
        entered = asyncio.Event()

        class Stuck(RecordingHandler):
            async def handle_condition(self, report: Any) -> None:
                entered.set()
                await release.wait()

        intake = AlertSubscriber(handler=Stuck())
        assert intake.stalled is False
        task = asyncio.create_task(
            intake._handle_message("alerts/critical", condition_payload())
        )
        await entered.wait()
        assert intake.stalled is False
        monkeypatch.setattr(subscriber, "STALL_LIMIT_S", 0.0)
        assert intake.stalled is True
        release.set()
        await task
        assert intake.stalled is False

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

    async def test_a_database_hiccup_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
        handler = RecordingHandler(failures=2)
        intake = AlertSubscriber(handler=handler)
        await intake._handle_message("alerts/critical", condition_payload())
        assert len(handler.conditions) == 1

    async def test_a_long_database_outage_keeps_the_message(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Paho acknowledges a QoS 1 message on receipt: a message given up is gone for
        # good, and the PLC snapshot never raises a lost activation again.
        monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
        handler = RecordingHandler(failures=5)
        intake = AlertSubscriber(handler=handler)
        before = counts()
        with caplog.at_level(logging.DEBUG, logger="alert_manager.subscriber"):
            await intake._handle_message("alerts/critical", condition_payload())
        assert len(handler.conditions) == 1
        assert delta(before)["failed"] == 0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "database unavailable" in warnings[0].getMessage()
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert "stored again after 6 attempts" in caplog.text

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionRefusedError("connect call failed"),
            TimeoutError(),
            InterfaceError("SELECT", {}, Exception("connection closed")),
        ],
    )
    async def test_transient_errors_are_retried(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
        handler = RecordingHandler(failures=2, error=error)
        intake = AlertSubscriber(handler=handler)
        await intake._handle_message("alerts/critical", condition_payload())
        assert len(handler.conditions) == 1

    @pytest.mark.parametrize(
        "error",
        [
            IntegrityError("INSERT", {}, Exception("duplicate key")),
            DataError("INSERT", {}, Exception("value out of range")),
        ],
    )
    async def test_a_permanent_database_error_is_not_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        error: Exception,
    ) -> None:
        monkeypatch.setattr(subscriber, "STORE_RETRY_DELAY_S", 0.0)
        handler = RecordingHandler(failures=1, error=error)
        intake = AlertSubscriber(handler=handler)
        before = counts()
        with caplog.at_level(logging.WARNING, logger="alert_manager.subscriber"):
            await intake._handle_message("alerts/critical", condition_payload())
        assert (delta(before)["failed"], handler.conditions) == (1, [])
        assert "could not be stored" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_the_retry_delay_backs_off_to_a_cap(self) -> None:
        delays = [subscriber.store_retry_delay_s(n) for n in range(1, 9)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
        assert subscriber.store_retry_delay_s(10_000) == 30.0

    async def test_an_unexpected_error_is_not_retried(self) -> None:
        handler = RecordingHandler(failures=1, error=RuntimeError("bug"))
        intake = AlertSubscriber(handler=handler)
        before = counts()
        await intake._handle_message("alerts/critical", condition_payload())
        assert (delta(before)["failed"], handler.conditions) == (1, [])

    async def test_hostile_payloads_are_rejected_and_counted_not_failed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        handler = RecordingHandler()
        intake = AlertSubscriber(handler=handler)
        before = counts()
        with caplog.at_level(logging.WARNING, logger="alert_manager.subscriber"):
            await intake._handle_message("alerts/critical", b"[" * 50_000)
            await intake._handle_message(
                "alerts/critical", condition_payload(value=10**400)
            )
            await intake._handle_message(
                "alerts/critical", condition_payload(timestamp_ms=1e300)
            )
        assert (delta(before)["invalid"], delta(before)["failed"]) == (3, 0)
        assert handler.conditions == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_a_payload_that_is_not_bytes_is_counted(
        self, broker: type[FakeBroker]
    ) -> None:
        broker.deliveries = [delivery("alerts/warning", "a text payload")]
        intake = AlertSubscriber("broker", 1883, RecordingHandler())
        before = rejected("not_bytes")
        task = asyncio.create_task(intake.run())
        try:
            await until(lambda: rejected("not_bytes") > before)
        finally:
            task.cancel()
        assert rejected("not_bytes") - before == 1

    async def test_without_a_handler_messages_are_skipped(self) -> None:
        intake = AlertSubscriber()
        before = counts()
        await intake._handle_message("alerts/critical", condition_payload())
        assert delta(before) == {
            "received": 1,
            "invalid": 0,
            "not_bytes": 0,
            "failed": 0,
        }

    async def test_end_to_end_into_the_processor(
        self, queries: AlarmQueries, processor: AlarmProcessor, recorder: Recorder
    ) -> None:
        intake = AlertSubscriber(handler=processor)
        await intake._handle_message("alerts/critical", condition_payload())
        alarm = await only_alarm(queries)
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

    def test_postgresql_connections_have_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without them a database that accepts connections but stops answering blocks
        # the processor lock, and with it the whole intake, forever.
        seen: dict[str, Any] = {}

        def capture(url: str, **options: Any) -> str:
            seen.update(options, url=url)
            return "engine"

        monkeypatch.setattr(db, "create_async_engine", capture)
        db.create_engine("postgresql+asyncpg://alarms@db/cogniboiler")
        assert seen["pool_pre_ping"] is True
        assert seen["hide_parameters"] is True
        assert seen["pool_timeout"] == db.POOL_TIMEOUT_S
        assert seen["connect_args"] == {
            "timeout": db.CONNECT_TIMEOUT_S,
            "command_timeout": db.COMMAND_TIMEOUT_S,
            "server_settings": {
                "statement_timeout": str(int(db.COMMAND_TIMEOUT_S * 1000))
            },
        }

    def test_a_file_database_gets_no_postgresql_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def capture(url: str, **options: Any) -> str:
            seen.update(options)
            return "engine"

        monkeypatch.setattr(db, "create_async_engine", capture)
        db.create_engine("sqlite+aiosqlite:///alarms.db")
        assert "connect_args" not in seen and "pool_timeout" not in seen

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
        monkeypatch.setattr(service, "create_engine", lambda: create_async_engine(url))
        with caplog.at_level(logging.ERROR, logger="alert_manager"):
            assert await service.main(self._args()) == 1
        assert "Alarm tables missing: alarm_events, alarm_transitions" in caplog.text

    async def test_it_wires_intake_processor_publisher_and_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"sqlite+aiosqlite:///{(tmp_path / 'alarms.db').as_posix()}"
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
        monkeypatch.setattr(service, "create_engine", lambda: create_async_engine(url))
        monkeypatch.setattr(service, "start_metrics_server", lambda port, host: None)
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
            healthy = True

            def __init__(self, host: str, port: int, handler: Any, **_: Any) -> None:
                assert isinstance(handler, AlarmProcessor)

            async def run(self) -> None:
                events.append("intake ran")

        class Server:
            async def stop(self, grace: float) -> None:
                events.append(f"server stopped {grace}")

        async def start(servicer: AlarmServicer, port: int) -> tuple[Server, int]:
            events.append(f"server started {port}")
            return Server(), 50053

        monkeypatch.setattr(service, "AlarmChangePublisher", Publisher)
        monkeypatch.setattr(service, "AlertSubscriber", Intake)
        monkeypatch.setattr(service, "start_server", start)
        assert await service.main(self._args()) == 0
        assert events[:3] == [
            "publisher alert-manager broker-pass",
            "publisher started",
            "server started 0",
        ]
        assert "intake ran" in events
        assert events[-2:] == ["server stopped 5", "publisher closed"]

    async def test_a_failing_cleanup_step_does_not_skip_the_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"sqlite+aiosqlite:///{(tmp_path / 'alarms.db').as_posix()}"
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()
        disposed: list[str] = []

        class Engine:
            def __init__(self) -> None:
                self._real = create_async_engine(url)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

            async def dispose(self) -> None:
                disposed.append("engine")
                await self._real.dispose()

        events: list[str] = []

        class Publisher:
            def __init__(self, *_: Any, **__: Any) -> None:
                return None

            def start(self) -> None:
                return None

            async def aclose(self) -> None:
                events.append("publisher closed")

        class Intake:
            connected = True
            healthy = True

            def __init__(self, *_: Any, **__: Any) -> None:
                return None

            async def run(self) -> None:
                return None

        class Server:
            async def stop(self, grace: float) -> None:
                raise RuntimeError("stop failed")

        async def start(servicer: AlarmServicer, port: int) -> tuple[Server, int]:
            return Server(), 50053

        monkeypatch.setattr(service, "create_engine", Engine)
        monkeypatch.setattr(service, "start_metrics_server", lambda port, host: None)
        monkeypatch.setattr(service, "AlarmChangePublisher", Publisher)
        monkeypatch.setattr(service, "AlertSubscriber", Intake)
        monkeypatch.setattr(service, "start_server", start)
        with pytest.raises(RuntimeError, match="stop failed"):
            await service.main(self._args())
        assert events == ["publisher closed"]
        assert disposed == ["engine"]

    def test_it_runs_under_the_shared_service_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []

        async def main(args: argparse.Namespace) -> int:
            ran.append(f"main {args.grpc_port}")
            return 3

        def run_service(entry_point: Any) -> int:
            code: int = asyncio.run(entry_point())
            return code

        monkeypatch.setattr(entry, "main", main)
        monkeypatch.setattr(entry, "run_service", run_service)
        monkeypatch.setattr(entry, "configure_logging", ran.append)
        assert entry.run(["--grpc-port", "50999"]) == 3
        assert ran == ["alert-manager", "main 50999"]

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
