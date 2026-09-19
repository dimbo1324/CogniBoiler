"""The realtime hub and its upstream sources."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from aiomqtt import MqttError
from api_gateway.realtime import sources
from api_gateway.realtime.hub import Channel, RealtimeHub, Subscriber, encode
from gateway_fakes import FakePhysicsClient, FakePLCClient, system_state


def frames(subscriber: Subscriber) -> list[dict[str, Any]]:
    decoded = []
    while not subscriber.queue.empty():
        decoded.append(json.loads(subscriber.queue.get_nowait()))
    return decoded


class TestSubscriber:
    def test_telemetry_is_limited_to_the_rate(self) -> None:
        subscriber = Subscriber(queue_size=8, max_rate_hz=2.0)
        assert subscriber.telemetry_due(100.0)
        assert not subscriber.telemetry_due(100.4)
        assert subscriber.telemetry_due(100.5)

    def test_a_zero_rate_is_unlimited(self) -> None:
        subscriber = Subscriber(queue_size=8, max_rate_hz=0.0)
        assert all(subscriber.telemetry_due(1.0) for _ in range(5))

    async def test_a_full_queue_drops_the_oldest_telemetry(self) -> None:
        subscriber = Subscriber(queue_size=2, max_rate_hz=10.0)
        for frame in ("a", "b", "c"):
            subscriber.offer_frame(frame)
        assert subscriber.dropped_frames == 1
        assert [subscriber.queue.get_nowait() for _ in range(2)] == ["b", "c"]
        assert subscriber.overflowed is False

    async def test_an_event_that_does_not_fit_marks_the_subscriber(self) -> None:
        subscriber = Subscriber(queue_size=1, max_rate_hz=10.0)
        subscriber.offer_event("first")
        subscriber.offer_event("second")
        assert subscriber.overflowed is True
        assert subscriber.queue.get_nowait() == "first"

    async def test_a_control_reply_is_compact_json(self) -> None:
        subscriber = Subscriber(queue_size=1, max_rate_hz=10.0)
        subscriber.send_control({"type": "pong", "ts_ms": 1})
        assert subscriber.queue.get_nowait() == '{"type":"pong","ts_ms":1}'

    def test_non_finite_numbers_are_not_encoded(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            encode({"value": float("nan")})


class TestHub:
    async def test_the_rate_is_clamped_between_a_tenth_and_the_server_cap(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        assert hub.register(50.0).max_rate_hz == 10.0
        assert hub.register(0.0).max_rate_hz == pytest.approx(0.1)
        assert hub.register().max_rate_hz == 10.0
        assert hub.subscriber_count == 3

    async def test_data_reaches_only_the_subscribers_of_its_channel(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        alarms, plc = hub.register(), hub.register()
        hub.subscribe(alarms, {Channel.ALARMS})
        hub.subscribe(plc, {Channel.PLC})
        hub.publish(Channel.ALARMS, "change", {"alarm_id": 7})
        (frame,) = frames(alarms)
        assert frame["type"] == "data"
        assert (frame["channel"], frame["kind"]) == ("alarms", "change")
        assert frame["data"] == {"alarm_id": 7}
        assert frames(plc) == []

    async def test_a_new_subscriber_starts_from_the_latest_snapshots(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        hub.publish(Channel.PLC, "status", {"mode": "auto"})
        hub.publish(Channel.PLC, "event", {"kind": "trip"})
        hub.publish(Channel.TELEMETRY, "state", {"t": 1})
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.PLC})
        received = frames(subscriber)
        assert [(f["kind"], f["data"]) for f in received] == [
            ("status", {"mode": "auto"})
        ]

    async def test_subscribing_again_does_not_repeat_the_snapshot(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        hub.publish(Channel.PLC, "status", {"mode": "auto"})
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.PLC})
        hub.subscribe(subscriber, {Channel.PLC, Channel.ALARMS})
        assert len(frames(subscriber)) == 1

    async def test_unsubscribing_and_unregistering(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.ALARMS, Channel.PLC})
        hub.unsubscribe(subscriber, {Channel.ALARMS})
        hub.publish(Channel.ALARMS, "change", {})
        assert frames(subscriber) == []
        hub.unregister(subscriber)
        assert hub.subscriber_count == 0

    async def test_telemetry_follows_each_subscriber_rate(self) -> None:
        hub = RealtimeHub(queue_size=16, max_rate_hz=10.0)
        subscriber = hub.register(1.0)
        hub.subscribe(subscriber, {Channel.TELEMETRY})
        for step in range(5):
            hub.publish(Channel.TELEMETRY, "state", {"step": step})
        assert [f["data"]["step"] for f in frames(subscriber)] == [0]

    async def test_the_telemetry_age_starts_unknown(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        assert hub.telemetry_age_s() is None
        hub.publish(Channel.TELEMETRY, "state", {})
        age = hub.telemetry_age_s()
        assert age is not None and 0.0 <= age < 1.0


async def first_frame(subscriber: Subscriber) -> dict[str, Any]:
    frame: dict[str, Any] = json.loads(
        await asyncio.wait_for(subscriber.queue.get(), timeout=2.0)
    )
    return frame


class TestSources:
    async def test_plant_states_become_telemetry(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.TELEMETRY})
        physics = FakePhysicsClient()
        physics.streamed = [system_state()]
        task = asyncio.create_task(sources.run_telemetry(hub, physics))  # type: ignore[arg-type]
        try:
            frame = await first_frame(subscriber)
        finally:
            task.cancel()
        assert frame["kind"] == "state"
        assert frame["data"]["boiler"]["pressure_pa"] == 140.0e5
        assert physics.calls[0] == ("stream_system_state", 0.0)

    async def test_the_plc_status_is_polled(self) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.PLC})
        task = asyncio.create_task(
            sources.run_plc_status(hub, FakePLCClient(), 0.01)  # type: ignore[arg-type]
        )
        try:
            first = await first_frame(subscriber)
            second = await first_frame(subscriber)
        finally:
            task.cancel()
        assert first["kind"] == second["kind"] == "status"
        assert first["data"]["mode"] == "auto"

    async def test_an_outage_is_logged_once_and_its_end_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(sources, "RECONNECT_DELAY_S", 0.001)
        hub = RealtimeHub(queue_size=64, max_rate_hz=10.0)
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.PLC})
        plc = FakePLCClient()
        plc.down = True
        with caplog.at_level(logging.INFO, logger="api_gateway.realtime.sources"):
            task = asyncio.create_task(
                sources.run_plc_status(hub, plc, 0.001)  # type: ignore[arg-type]
            )
            await asyncio.sleep(0.05)
            plc.down = False
            try:
                await first_frame(subscriber)
            finally:
                task.cancel()
        levels = [
            (record.levelno, record.getMessage().split(":")[0])
            for record in caplog.records
        ]
        assert levels == [
            (logging.WARNING, "Realtime source plc status unavailable"),
            (logging.INFO, "Realtime source plc status recovered"),
        ]

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b'{"alarm_id": 7}', {"alarm_id": 7}),
            (bytearray(b'{"a": 1}'), {"a": 1}),
            (b"[1, 2]", None),
            (b"not json", None),
            (b"\xff\xfe", None),
            ("text", None),
            (None, None),
        ],
    )
    def test_only_json_objects_are_forwarded(
        self, payload: object, expected: dict[str, Any] | None
    ) -> None:
        assert sources._json_object(payload) == expected


class FakeMqttClient:
    """aiomqtt.Client stand-in: the messages it delivers, then a lost connection."""

    deliveries: list[tuple[str, bytes]] = []
    subscribed: list[str] = []
    connections = 0

    def __init__(self, **_: object) -> None:
        FakeMqttClient.connections += 1

    async def __aenter__(self) -> FakeMqttClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def subscribe(self, topic: str, qos: int) -> None:
        FakeMqttClient.subscribed.append(topic)

    @property
    def messages(self) -> AsyncIterator[SimpleNamespace]:
        async def deliver() -> AsyncIterator[SimpleNamespace]:
            for topic, payload in FakeMqttClient.deliveries:
                yield SimpleNamespace(topic=topic, payload=payload)
            raise MqttError("connection lost")

        return deliver()


class TestMqttEvents:
    async def test_plc_events_and_alarm_changes_are_routed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMqttClient.deliveries = [
            ("plc/events", b'{"kind": "trip"}'),
            ("alarms/changes", b'{"alarm_id": 7}'),
            ("alarms/changes", b"garbage"),
            ("other/topic", b'{"x": 1}'),
        ]
        FakeMqttClient.subscribed = []
        monkeypatch.setattr(sources, "Client", FakeMqttClient)
        monkeypatch.setattr(sources, "RECONNECT_DELAY_S", 60.0)
        hub = RealtimeHub(queue_size=8, max_rate_hz=10.0)
        subscriber = hub.register()
        hub.subscribe(subscriber, {Channel.PLC, Channel.ALARMS})
        task = asyncio.create_task(sources.run_mqtt_events(hub, "broker", 1883))
        try:
            plc_event = await first_frame(subscriber)
            alarm_change = await first_frame(subscriber)
        finally:
            task.cancel()
        assert (plc_event["channel"], plc_event["kind"]) == ("plc", "event")
        assert (alarm_change["channel"], alarm_change["kind"]) == ("alarms", "change")
        assert alarm_change["data"] == {"alarm_id": 7}
        assert subscriber.queue.empty()
        assert FakeMqttClient.subscribed == ["plc/events", "alarms/changes"]

    async def test_a_lost_broker_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMqttClient.deliveries = []
        FakeMqttClient.connections = 0
        monkeypatch.setattr(sources, "Client", FakeMqttClient)
        monkeypatch.setattr(sources, "RECONNECT_DELAY_S", 0.001)
        hub = RealtimeHub(queue_size=8, max_rate_hz=10.0)
        task = asyncio.create_task(sources.run_mqtt_events(hub, "broker", 1883))
        try:
            async with asyncio.timeout(5.0):
                while FakeMqttClient.connections < 2:
                    await asyncio.sleep(0.001)
        finally:
            task.cancel()
        assert FakeMqttClient.connections >= 2
