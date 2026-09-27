"""The queue-backed publisher: in order, at least once, bounded, and honest on the way out."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
from cogniboiler_runtime.mqtt import MqttSession
from cogniboiler_runtime.mqtt_queue import QueuedMessage, QueuedMqttPublisher

type Sent = tuple[str, bytes, int, bool]


class FakeClient:
    """A broker link that records what was published and can drop mid-stream."""

    def __init__(self, fail_on_publish: int | None = None) -> None:
        self.sent: list[Sent] = []
        self._fail_on = fail_on_publish
        self._attempts = 0

    async def publish(
        self, topic: str, payload: bytes, *, qos: int, retain: bool
    ) -> None:
        self._attempts += 1
        if self._fail_on is not None and self._attempts == self._fail_on:
            raise OSError("connection lost while publishing")
        self.sent.append((topic, payload, qos, retain))


class Broker:
    """Hands out one client per connection; an exception in `plan` refuses it."""

    def __init__(self, *plan: Exception | FakeClient | None) -> None:
        self._plan = iter(plan)
        self.clients: list[FakeClient] = []

    def open(self) -> AbstractAsyncContextManager[FakeClient]:
        @asynccontextmanager
        async def connect() -> AsyncIterator[FakeClient]:
            step = next(self._plan, None)
            if isinstance(step, Exception):
                raise step
            client = step or FakeClient()
            self.clients.append(client)
            yield client

        return connect()

    @property
    def sent(self) -> list[Sent]:
        return [message for client in self.clients for message in client.sent]


def publisher_for(broker: Broker, **options: object) -> QueuedMqttPublisher[FakeClient]:
    session: MqttSession[FakeClient] = MqttSession(
        broker.open, name="test", reconnect_delay_s=0.0
    )
    return QueuedMqttPublisher(session, name="test publisher", **options)  # type: ignore[arg-type]


async def until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5.0):
        while not condition():
            await asyncio.sleep(0)


def message(number: int, topic: str = "plc/events") -> QueuedMessage:
    return QueuedMessage(topic, f'{{"n": {number}}}'.encode())


class TestOrderAndDelivery:
    async def test_messages_go_out_in_the_order_they_were_queued(self) -> None:
        broker = Broker()
        publisher = publisher_for(broker)
        publisher.start()
        for number in range(3):
            publisher.enqueue(message(number))
        await until(lambda: len(broker.sent) == 3)
        await publisher.aclose()
        assert [payload for _, payload, _, _ in broker.sent] == [
            b'{"n": 0}',
            b'{"n": 1}',
            b'{"n": 2}',
        ]
        assert all((qos, retain) == (1, False) for _, _, qos, retain in broker.sent)

    async def test_what_was_queued_while_the_broker_was_away_goes_out_in_order(
        self,
    ) -> None:
        broker = Broker(OSError("refused"), OSError("refused"), None)
        publisher = publisher_for(broker)
        for number in range(3):
            publisher.enqueue(message(number))
        publisher.start()
        await until(lambda: len(broker.sent) == 3)
        await publisher.aclose()
        assert [payload for _, payload, _, _ in broker.sent] == [
            b'{"n": 0}',
            b'{"n": 1}',
            b'{"n": 2}',
        ]

    async def test_a_publish_that_fails_is_sent_again_on_the_next_connection(
        self,
    ) -> None:
        published: list[str] = []
        failed: list[str] = []
        broker = Broker(FakeClient(fail_on_publish=2), None)
        publisher = publisher_for(
            broker, on_published=published.append, on_publish_error=failed.append
        )
        for number in range(3):
            publisher.enqueue(message(number, topic=f"t/{number}"))
        publisher.start()
        await until(lambda: len(broker.sent) == 3)
        await publisher.aclose()
        first, second = broker.clients
        assert [topic for topic, *_ in first.sent] == ["t/0"]
        assert [topic for topic, *_ in second.sent] == ["t/1", "t/2"]
        assert failed == ["t/1"]
        assert published == ["t/0", "t/1", "t/2"]
        assert publisher.pending == 0

    async def test_retain_and_qos_travel_with_the_message(self) -> None:
        broker = Broker()
        publisher = publisher_for(broker)
        publisher.start()
        publisher.enqueue(QueuedMessage("status/x", b"state", qos=0, retain=True))
        await until(lambda: len(broker.sent) == 1)
        await publisher.aclose()
        assert broker.sent == [("status/x", b"state", 0, True)]


class TestABoundedQueue:
    async def test_a_full_queue_drops_the_oldest_and_counts_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        broker = Broker()
        publisher = publisher_for(broker, limit=2)
        with caplog.at_level(logging.ERROR, logger="cogniboiler_runtime.mqtt_queue"):
            for number in range(4):
                publisher.enqueue(message(number))
        assert publisher.dropped == 2
        assert publisher.pending == 2
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "test publisher" in errors[0].getMessage()

        publisher.start()
        await until(lambda: len(broker.sent) == 2)
        await publisher.aclose()
        assert [payload for _, payload, _, _ in broker.sent] == [
            b'{"n": 2}',
            b'{"n": 3}',
        ]

    async def test_drops_are_logged_at_the_first_and_every_hundredth(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        publisher = publisher_for(Broker(), limit=1)
        with caplog.at_level(logging.ERROR, logger="cogniboiler_runtime.mqtt_queue"):
            for number in range(202):
                publisher.enqueue(message(number))
        assert publisher.dropped == 201
        assert len(caplog.records) == 3

    def test_a_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            publisher_for(Broker(), limit=0)


class TestAvailability:
    async def test_online_on_connect_and_offline_before_stopping(self) -> None:
        broker = Broker()
        publisher = publisher_for(broker, availability_topic="status/plc-controller")
        publisher.start()
        await until(lambda: publisher.connected and len(broker.sent) == 1)
        publisher.enqueue(message(1))
        await publisher.aclose()
        assert broker.sent == [
            ("status/plc-controller", b"online", 1, True),
            ("plc/events", b'{"n": 1}', 1, False),
            ("status/plc-controller", b"offline", 1, True),
        ]

    async def test_without_a_topic_nothing_is_announced(self) -> None:
        broker = Broker()
        publisher = publisher_for(broker)
        publisher.start()
        await until(lambda: publisher.connected)
        await publisher.aclose()
        assert broker.sent == []


class TestLifecycle:
    async def test_closing_a_publisher_that_never_started_is_a_no_op(self) -> None:
        publisher = publisher_for(Broker(), availability_topic="status/x")
        await publisher.aclose()
        assert publisher.pending == 0

    async def test_closing_while_the_broker_is_away_does_not_wait_for_it(
        self,
    ) -> None:
        broker = Broker(*(OSError("refused") for _ in range(10_000)))
        publisher = publisher_for(broker, availability_topic="status/x")
        publisher.start()
        publisher.enqueue(message(1))
        async with asyncio.timeout(5.0):
            await publisher.aclose(drain_timeout_s=3600.0)
        assert publisher.connected is False

    async def test_start_twice_runs_one_publisher(self) -> None:
        broker = Broker()
        publisher = publisher_for(broker)
        publisher.start()
        publisher.start()
        await until(lambda: publisher.connected)
        await publisher.aclose()
        assert len(broker.clients) == 1


class TestPeriodicMessage:
    async def test_it_runs_once_the_connection_is_up(self) -> None:
        broker = Broker()
        calls: list[FakeClient] = []

        async def snapshot(client: FakeClient) -> None:
            calls.append(client)

        publisher = publisher_for(broker, periodic=snapshot, periodic_interval_s=3600.0)
        publisher.start()
        await until(lambda: len(calls) == 1)
        await publisher.aclose()
        assert calls == broker.clients

    async def test_it_repeats_at_its_interval(self) -> None:
        calls = 0

        async def snapshot(client: FakeClient) -> None:
            nonlocal calls
            calls += 1

        publisher = publisher_for(
            Broker(), periodic=snapshot, periodic_interval_s=0.001
        )
        publisher.start()
        await until(lambda: calls >= 3)
        await publisher.aclose()

    def test_a_periodic_message_needs_a_positive_interval(self) -> None:
        async def snapshot(client: FakeClient) -> None:
            return None

        with pytest.raises(ValueError, match="periodic_interval_s"):
            publisher_for(Broker(), periodic=snapshot)
        with pytest.raises(ValueError, match="periodic_interval_s"):
            publisher_for(Broker(), periodic=snapshot, periodic_interval_s=0.0)
