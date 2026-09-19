"""What the PLC publishes over MQTT, and the alarm condition monitor."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from aiomqtt import MqttError
from plc_controller import events
from plc_controller.alarms import (
    AlarmCondition,
    AlarmConditionMonitor,
    AlarmTransition,
    Arming,
    ConditionRule,
    Direction,
    Severity,
)
from plc_controller.events import (
    TOPIC_ALERT_CRITICAL,
    TOPIC_ALERT_SNAPSHOT,
    TOPIC_ALERT_WARNING,
    TOPIC_AVAILABILITY,
    TOPIC_PLC_EVENTS,
    PlcEvent,
    PlcEventKind,
    PlcPublisher,
)
from plc_controller.safety import ArmingState

LEVEL_LOW = ConditionRule(
    "water_level_m", "m", Severity.CRITICAL, Direction.LOW, 1.0, 0.2
)
STACK_HIGH = ConditionRule(
    "stack_temp_k", "K", Severity.WARNING, Direction.HIGH, 450.0, 5.0, Arming.FIRING
)
ARMED = ArmingState(on_line=True, firing_proven=True)


def transition(
    rule: ConditionRule, active: bool = True, at: int = 5
) -> AlarmTransition:
    return AlarmTransition(AlarmCondition(rule, 0.8, at), active, at)


class Broker:
    published: list[tuple[str, Any, int, bool]] = []
    connections: list[dict[str, Any]] = []
    failures = 0

    def __init__(self, **options: Any) -> None:
        Broker.connections.append(options)

    async def __aenter__(self) -> Broker:
        if Broker.failures:
            Broker.failures -= 1
            raise MqttError("broker unreachable")
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def publish(
        self, topic: str, payload: Any = None, qos: int = 0, retain: bool = False
    ) -> None:
        Broker.published.append((topic, payload, qos, retain))


@pytest.fixture(autouse=True)
def broker(monkeypatch: pytest.MonkeyPatch) -> type[Broker]:
    Broker.published = []
    Broker.connections = []
    Broker.failures = 0
    monkeypatch.setattr(events, "Client", Broker)
    monkeypatch.setattr(events, "RECONNECT_DELAY_S", 0.001)
    return Broker


async def until(predicate: Any) -> None:
    async with asyncio.timeout(5.0):
        while not predicate():
            await asyncio.sleep(0.001)


def topics() -> list[str]:
    return [topic for topic, *_ in Broker.published]


class TestPublisher:
    async def test_online_then_snapshot_then_messages_in_order(self) -> None:
        monitor = AlarmConditionMonitor((LEVEL_LOW,))
        monitor.evaluate({"water_level_m": 0.5}, ARMED, 1)
        publisher = PlcPublisher(
            "broker",
            1883,
            active_conditions=monitor.active,
            username="plc-controller",
            password="pw",
        )
        publisher.publish_alarm(transition(LEVEL_LOW))
        publisher.publish_alarm(transition(STACK_HIGH, active=False))
        publisher.publish_event(
            PlcEvent(PlcEventKind.MODE_CHANGED, "operator1", {"to": "manual"}, 9)
        )
        publisher.start()
        publisher.start()
        await until(lambda: len(Broker.published) >= 5)
        await publisher.aclose()
        assert topics()[:5] == [
            TOPIC_AVAILABILITY,
            TOPIC_ALERT_CRITICAL,
            TOPIC_ALERT_WARNING,
            TOPIC_PLC_EVENTS,
            TOPIC_ALERT_SNAPSHOT,
        ]
        assert Broker.published[0][1:] == ("online", 1, True)
        critical = json.loads(Broker.published[1][1])
        assert critical["state"] == "active"
        assert critical["key"] == "plc-controller:water_level_m:low:critical"
        assert critical["action"] == "emergency_stop"
        assert json.loads(Broker.published[2][1])["state"] == "cleared"
        event = json.loads(Broker.published[3][1])
        assert (event["kind"], event["operator_id"], event["detail"]) == (
            "mode_changed",
            "operator1",
            {"to": "manual"},
        )
        assert event["event_id"] == "mode_changed:9"
        snapshot = json.loads(Broker.published[4][1])
        assert snapshot["active_keys"] == ["plc-controller:water_level_m:low:critical"]
        assert topics()[-1] == TOPIC_AVAILABILITY
        assert Broker.published[-1][1:] == (b"offline", 1, True)
        options = Broker.connections[0]
        assert (options["username"], options["identifier"]) == (
            "plc-controller",
            "plc-controller",
        )
        assert options["will"].retain is True

    async def test_messages_wait_for_the_broker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        Broker.failures = 3
        publisher = PlcPublisher("broker", 1883)
        publisher.publish_event(PlcEvent(PlcEventKind.ESTOP_RESET, "eng"))
        with caplog.at_level(logging.INFO, logger="plc_controller.events"):
            publisher.start()
            await until(lambda: TOPIC_PLC_EVENTS in topics())
        await publisher.aclose()
        assert len(Broker.connections) == 4
        assert caplog.text.count("PLC publisher lost MQTT") == 1
        assert "PLC publisher reconnected" in caplog.text

    async def test_a_full_queue_drops_the_oldest(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(events, "QUEUE_LIMIT", 2)
        publisher = PlcPublisher("broker", 1883)
        with caplog.at_level(logging.ERROR, logger="plc_controller.events"):
            for number in range(4):
                publisher.publish_event(
                    PlcEvent(PlcEventKind.RUN_CHANGED, "physics-engine", {"n": number})
                )
        assert publisher.dropped == 2
        assert caplog.text.count("PLC publish queue full") == 1
        publisher.start()
        await until(lambda: topics().count(TOPIC_PLC_EVENTS) == 2)
        await publisher.aclose()
        numbers = [
            json.loads(payload)["detail"]["n"]
            for topic, payload, *_ in Broker.published
            if topic == TOPIC_PLC_EVENTS
        ]
        assert numbers == [2, 3]

    async def test_a_disabled_publisher_does_nothing(self) -> None:
        publisher = PlcPublisher("broker", 1883, enabled=False)
        publisher.publish_event(PlcEvent(PlcEventKind.MANUAL_TRIP, "eng"))
        publisher.start()
        await publisher.aclose()
        assert Broker.connections == []

    async def test_the_snapshot_repeats_while_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(events, "SNAPSHOT_INTERVAL_S", 0.01)
        publisher = PlcPublisher("broker", 1883)
        publisher.start()
        await until(lambda: topics().count(TOPIC_ALERT_SNAPSHOT) >= 3)
        await publisher.aclose()


class TestMonitor:
    def test_a_condition_rises_holds_and_clears_past_its_deadband(self) -> None:
        monitor = AlarmConditionMonitor((LEVEL_LOW,))
        (raised,) = monitor.evaluate({"water_level_m": 0.9}, ARMED, 10)
        assert (raised.active, raised.condition.value) == (True, 0.9)
        assert monitor.evaluate({"water_level_m": 1.1}, ARMED, 20) == []
        assert monitor.active()[0].value == 1.1
        (cleared,) = monitor.evaluate({"water_level_m": 1.3}, ARMED, 30)
        assert (cleared.active, cleared.condition.since_ms) == (False, 10)
        assert monitor.active() == ()

    def test_an_unarmed_condition_neither_rises_nor_stays(self) -> None:
        monitor = AlarmConditionMonitor((STACK_HIGH,))
        idle = ArmingState(on_line=False, firing_proven=False)
        assert monitor.evaluate({"stack_temp_k": 500.0}, idle, 1) == []
        monitor.evaluate({"stack_temp_k": 500.0}, ARMED, 2)
        (dropped,) = monitor.evaluate({"stack_temp_k": 500.0}, idle, 3)
        assert dropped.active is False

    def test_missing_and_non_finite_values_are_ignored(self) -> None:
        monitor = AlarmConditionMonitor((LEVEL_LOW,))
        assert monitor.evaluate({}, ARMED, 1) == []
        assert monitor.evaluate({"water_level_m": float("nan")}, ARMED, 1) == []

    def test_critical_conditions_come_first_then_the_oldest(self) -> None:
        monitor = AlarmConditionMonitor((STACK_HIGH, LEVEL_LOW))
        monitor.evaluate({"stack_temp_k": 500.0}, ARMED, 1)
        monitor.evaluate({"water_level_m": 0.5}, ARMED, 2)
        assert [c.rule.parameter for c in monitor.active()] == [
            "water_level_m",
            "stack_temp_k",
        ]
        assert [c.rule.parameter for c in monitor.active_critical()] == [
            "water_level_m"
        ]

    def test_a_new_run_clears_every_condition(self) -> None:
        monitor = AlarmConditionMonitor((STACK_HIGH, LEVEL_LOW))
        monitor.evaluate({"stack_temp_k": 500.0, "water_level_m": 0.5}, ARMED, 1)
        cleared = monitor.clear_all(9)
        assert {(t.active, t.timestamp_ms) for t in cleared} == {(False, 9)}
        assert monitor.active() == ()

    def test_rule_keys_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique keys"):
            AlarmConditionMonitor((LEVEL_LOW, LEVEL_LOW))

    def test_the_message_names_value_and_limit(self) -> None:
        condition = AlarmCondition(STACK_HIGH, 461.25, 1)
        assert (
            condition.message
            == "stack_temp_k high warning: 461.2 K against limit 450 K"
        )
        assert STACK_HIGH.action == "warn"
