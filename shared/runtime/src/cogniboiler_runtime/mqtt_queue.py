"""Ordered, must-not-vanish-silently MQTT messages over one persistent connection.

plc-controller and alert-manager each carried the same publisher: a bounded deque that
drops its oldest message when full and says so, a wake-up event, and a drain loop that
publishes the head of the queue and only then removes it. The copies had drifted (one
drained with a deadline on close, the other not; neither counted failed publishes). This
is that publisher once, with both behaviours.

Delivery is at least once and in order: a message leaves the queue only after its publish
returned, so a publish that fails ends the session with the message still at the head,
and it goes out first on the next connection. A long outage costs the oldest messages,
never unbounded memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from cogniboiler_runtime.mqtt import MqttSession

DEFAULT_QUEUE_LIMIT = 1000
DEFAULT_CLOSE_DRAIN_TIMEOUT_S = 1.0
DROP_LOG_EVERY = 100
AVAILABILITY_QOS = 1

_LOGGER = logging.getLogger(__name__)


class SupportsPublish(Protocol):
    """The part of an MQTT client the publisher needs."""

    async def publish(
        self, topic: str, payload: bytes, *, qos: int, retain: bool
    ) -> object: ...


@dataclass(frozen=True)
class QueuedMessage:
    topic: str
    payload: bytes
    qos: int = 1
    retain: bool = False


class QueuedMqttPublisher[ClientT: SupportsPublish]:
    """A bounded, ordered publish queue drained by one `MqttSession`.

    With `availability_topic`, the publisher sends a retained "online" on every connect
    and queues a retained "offline" in `aclose`, before it stops — the MQTT will only
    covers a connection that dies, not a clean shutdown. With `periodic`, that coroutine
    runs on every connect and then every `periodic_interval_s` while connected.

    `on_published` and `on_publish_error` receive the topic of each publish that went
    out or failed; a service passes its metrics counters there.
    """

    def __init__(
        self,
        session: MqttSession[ClientT],
        *,
        name: str,
        limit: int = DEFAULT_QUEUE_LIMIT,
        availability_topic: str | None = None,
        periodic: Callable[[ClientT], Awaitable[None]] | None = None,
        periodic_interval_s: float | None = None,
        on_published: Callable[[str], None] | None = None,
        on_publish_error: Callable[[str], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if periodic is not None and (
            periodic_interval_s is None or periodic_interval_s <= 0.0
        ):
            raise ValueError("periodic_interval_s must be > 0 when periodic is set")
        self._session = session
        self._name = name
        self._limit = limit
        self._availability_topic = availability_topic
        self._periodic = periodic
        self._periodic_interval_s = periodic_interval_s or 0.0
        self._on_published = on_published
        self._on_publish_error = on_publish_error
        self._logger = logger or _LOGGER
        self._queue: deque[QueuedMessage] = deque()
        self._wakeup = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0

    @property
    def connected(self) -> bool:
        return self._session.connected

    @property
    def dropped(self) -> int:
        """Messages lost to a full queue since this publisher was made."""
        return self._dropped

    @property
    def pending(self) -> int:
        """Messages waiting to be published."""
        return len(self._queue)

    def enqueue(self, message: QueuedMessage) -> None:
        """Queue a message; never blocks, drops the oldest one when the queue is full."""
        if len(self._queue) >= self._limit:
            self._queue.popleft()
            self._dropped += 1
            if self._dropped == 1 or self._dropped % DROP_LOG_EVERY == 0:
                self._logger.error(
                    "%s: queue full, %d messages dropped so far",
                    self._name,
                    self._dropped,
                )
        self._queue.append(message)
        self._drained.clear()
        self._wakeup.set()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._session.run(self._work), name=f"{self._name} MQTT queue"
        )

    async def aclose(
        self, drain_timeout_s: float = DEFAULT_CLOSE_DRAIN_TIMEOUT_S
    ) -> None:
        """Queue "offline" if configured, give the queue a moment to drain, then stop.

        The wait happens only while connected: with the broker away there is nothing
        to drain into, and a shutdown must not hang on it.
        """
        task = self._task
        if task is None:
            return
        if self._availability_topic is not None:
            self.enqueue(
                QueuedMessage(
                    self._availability_topic,
                    b"offline",
                    qos=AVAILABILITY_QOS,
                    retain=True,
                )
            )
        if self._session.connected and drain_timeout_s > 0.0:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(drain_timeout_s):
                    await self._drained.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None

    async def _publish(self, client: ClientT, message: QueuedMessage) -> None:
        try:
            await client.publish(
                message.topic, message.payload, qos=message.qos, retain=message.retain
            )
        except Exception:
            if self._on_publish_error is not None:
                self._on_publish_error(message.topic)
            raise
        if self._on_published is not None:
            self._on_published(message.topic)

    async def _work(self, client: ClientT) -> None:
        if self._availability_topic is not None:
            await self._publish(
                client,
                QueuedMessage(
                    self._availability_topic,
                    b"online",
                    qos=AVAILABILITY_QOS,
                    retain=True,
                ),
            )
        loop = asyncio.get_running_loop()
        next_periodic = loop.time()
        while True:
            while self._queue:
                await self._publish(client, self._queue[0])
                self._queue.popleft()
            self._drained.set()
            if self._periodic is not None and loop.time() >= next_periodic:
                await self._periodic(client)
                next_periodic = loop.time() + self._periodic_interval_s
            self._wakeup.clear()
            if self._queue:
                continue
            timeout = (
                None
                if self._periodic is None
                else max(next_periodic - loop.time(), 0.0)
            )
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(timeout):
                    await self._wakeup.wait()
