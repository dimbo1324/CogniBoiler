"""Every JSON payload on the MQTT contract, from its producer through every consumer.

Producers and consumers live in different services and each side used to spell the keys
out by hand: renaming `to_state` in alert-manager would have broken the console's alarm
list and the historian's `alarm_changes` tag with no failing test. Here the real producer
function writes the bytes and the real consumer code reads them, in one process, so a
rename on either side fails the gate.

    plc-controller  alerts/warning, alerts/critical, alerts/snapshot  -> alert-manager
    plc-controller  plc/events      -> api-gateway (/ws plc), historian
    alert-manager   alarms/changes  -> api-gateway (/ws alarms), historian, opcua-server
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from aiomqtt import MqttError
from alert_manager.lifecycle import AlarmState
from alert_manager.payloads import change_payload, parse_condition, parse_snapshot
from alert_manager.views import AlarmView, TransitionView
from api_gateway.realtime import sources
from api_gateway.realtime.hub import Channel, RealtimeHub
from cogniboiler_runtime.contracts import ALARM_STATES
from cogniboiler_runtime.topics import (
    TOPIC_ALARM_CHANGES,
    TOPIC_ALERT_CRITICAL,
    TOPIC_ALERT_WARNING,
    TOPIC_PLC_EVENTS,
)
from historian.subscriber import HistorianSubscriber
from opcua_server.subscriber import MQTTOPCBridge
from plc_controller.alarms import (
    AlarmCondition,
    AlarmTransition,
    ConditionRule,
    Direction,
    Severity,
)
from plc_controller.events import (
    PlcEvent,
    PlcEventKind,
    alarm_payload,
    alarm_topic,
    event_payload,
    snapshot_payload,
)

AT_MS = 1_741_000_000_000


def plc_transition(severity: Severity, *, active: bool) -> AlarmTransition:
    rule = ConditionRule(
        "water_level_m", "m", severity, Direction.LOW, 3.5, deadband=0.05
    )
    return AlarmTransition(AlarmCondition(rule, 3.1, AT_MS - 2_000), active, AT_MS)


def plc_event() -> PlcEvent:
    return PlcEvent(
        PlcEventKind.MANUAL_TRIP,
        "operator1",
        {"reason": "drill", "fuel_valve": 0.0},
        AT_MS,
    )


def alarm_change() -> bytes:
    alarm = AlarmView(
        id=7,
        key="plc-controller:water_level_m:low:critical",
        source_service="plc-controller",
        parameter="water_level_m",
        severity="critical",
        direction="low",
        unit="m",
        state=AlarmState.ACTIVE_ACK,
        message="Drum level low-low",
        action="emergency_stop",
        topic=TOPIC_ALERT_CRITICAL,
        value=3.1,
        threshold=3.5,
        raised_at_ms=AT_MS - 2_000,
        cleared_at_ms=None,
        acknowledged_at_ms=AT_MS,
        acknowledged_by="operator1",
        ack_comment="seen",
        occurrence_count=1,
        updated_at_ms=AT_MS,
    )
    transition = TransitionView(
        id=3,
        alarm_id=7,
        from_state=AlarmState.ACTIVE_UNACK,
        to_state=AlarmState.ACTIVE_ACK,
        at_ms=AT_MS,
        actor="operator1",
        comment="seen",
        value=None,
    )
    return change_payload(alarm, transition)


class OneShotBroker:
    """aiomqtt.Client stand-in: delivers its messages once, then the link drops."""

    deliveries: list[tuple[str, bytes]] = []

    def __init__(self, **_: object) -> None:
        return None

    async def __aenter__(self) -> OneShotBroker:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def subscribe(self, topic: str, qos: int) -> None:
        return None

    @property
    def messages(self) -> AsyncIterator[SimpleNamespace]:
        async def deliver() -> AsyncIterator[SimpleNamespace]:
            for topic, payload in OneShotBroker.deliveries:
                yield SimpleNamespace(topic=topic, payload=payload)
            raise MqttError("connection lost")

        return deliver()


async def gateway_frames(
    monkeypatch: pytest.MonkeyPatch, topic: str, payload: bytes
) -> list[dict[str, Any]]:
    """What the gateway's realtime source sends to the browser for one message."""
    OneShotBroker.deliveries = [(topic, payload)]
    monkeypatch.setattr(sources, "Client", OneShotBroker)
    monkeypatch.setattr(sources, "RECONNECT_DELAY_S", 60.0)
    hub = RealtimeHub(queue_size=8, max_rate_hz=10.0)
    subscriber = hub.register()
    hub.subscribe(subscriber, {Channel.PLC, Channel.ALARMS})
    task = asyncio.create_task(sources.run_mqtt_events(hub, "broker", 1883))
    try:
        frame = await asyncio.wait_for(subscriber.queue.get(), timeout=2.0)
    finally:
        task.cancel()
    frames = [json.loads(frame)]
    while not subscriber.queue.empty():
        frames.append(json.loads(subscriber.queue.get_nowait()))
    return frames


class LineRecorder:
    """The historian's InfluxDB writer, keeping line protocol instead of writing."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_points(self, points: list[Any]) -> int:
        self.lines.extend(point.to_line_protocol() for point in points)
        return len(points)


async def historian_lines(topic: str, payload: bytes) -> list[str]:
    writer = LineRecorder()
    subscriber = HistorianSubscriber(writer)  # type: ignore[arg-type]
    await subscriber._handle_message(topic, payload)
    return writer.lines


class SilentOPC:
    async def update_variable(self, *_: object, **__: object) -> None:
        raise AssertionError("an alarm change writes no plant variable")


@pytest.fixture(autouse=True)
def fresh_broker() -> Iterator[None]:
    yield
    OneShotBroker.deliveries = []


class TestPlcToAlertManager:
    @pytest.mark.parametrize("severity", list(Severity))
    @pytest.mark.parametrize("active", [True, False])
    def test_a_condition_reaches_the_alarm_intake_whole(
        self, severity: Severity, active: bool
    ) -> None:
        transition = plc_transition(severity, active=active)
        topic = alarm_topic(transition)
        report = parse_condition(topic, alarm_payload(transition))
        rule = transition.condition.rule
        assert topic == (
            TOPIC_ALERT_CRITICAL
            if severity is Severity.CRITICAL
            else TOPIC_ALERT_WARNING
        )
        assert report.key == rule.key
        assert report.active is active
        assert (report.source_service, report.parameter) == (
            "plc-controller",
            "water_level_m",
        )
        assert (report.severity, report.direction, report.unit) == (
            severity.value,
            "low",
            "m",
        )
        assert (report.value, report.threshold) == (3.1, 3.5)
        assert report.action == rule.action
        assert report.message == transition.condition.message
        assert (report.topic, report.timestamp_ms) == (topic, AT_MS)

    def test_a_snapshot_reaches_the_alarm_intake_whole(self) -> None:
        active = [
            plc_transition(Severity.WARNING, active=True).condition,
            plc_transition(Severity.CRITICAL, active=True).condition,
        ]
        report = parse_snapshot(snapshot_payload(active, AT_MS))
        assert report.source_service == "plc-controller"
        assert report.active_keys == {condition.key for condition in active}
        assert report.timestamp_ms == AT_MS


class TestPlcEvents:
    async def test_the_gateway_forwards_the_event_as_published(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = event_payload(plc_event())
        (frame,) = await gateway_frames(monkeypatch, TOPIC_PLC_EVENTS, raw)
        assert (frame["channel"], frame["kind"]) == ("plc", "event")
        assert frame["data"] == json.loads(raw)
        assert frame["data"]["operator_id"] == "operator1"

    async def test_the_historian_records_kind_operator_and_detail(self) -> None:
        (line,) = await historian_lines(TOPIC_PLC_EVENTS, event_payload(plc_event()))
        assert line.startswith("plc_events,kind=manual_trip ")
        assert 'operator_id="operator1"' in line
        assert '\\"reason\\": \\"drill\\"' in line
        assert 'text="PLC manual_trip by operator1"' in line
        assert line.endswith(f" {AT_MS * 1_000_000}")


class TestAlarmChanges:
    async def test_the_gateway_forwards_the_change_as_published(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = alarm_change()
        (frame,) = await gateway_frames(monkeypatch, TOPIC_ALARM_CHANGES, raw)
        assert (frame["channel"], frame["kind"]) == ("alarms", "change")
        assert frame["data"] == json.loads(raw)

    async def test_the_historian_records_severity_state_and_actor(self) -> None:
        (line,) = await historian_lines(TOPIC_ALARM_CHANGES, alarm_change())
        assert line.startswith(
            "alarm_changes,parameter=water_level_m,severity=critical,state=ACTIVE_ACK "
        )
        assert "alarm_id=7" in line and 'actor="operator1"' in line
        assert "value=3.1" in line and "threshold=3.5" in line
        assert line.endswith(f" {AT_MS * 1_000_000}")

    async def test_the_opc_ua_projection_wakes_its_alarm_folder(self) -> None:
        changed = asyncio.Event()
        bridge = MQTTOPCBridge(SilentOPC(), alarms_changed=changed)  # type: ignore[arg-type]
        await bridge._handle_message(TOPIC_ALARM_CHANGES, alarm_change())
        assert changed.is_set()
        assert bridge.stats == {"received": 1, "mapped": 1, "skipped": 0}

    def test_the_contract_knows_every_lifecycle_state(self) -> None:
        assert set(ALARM_STATES) == {state.value for state in AlarmState}
