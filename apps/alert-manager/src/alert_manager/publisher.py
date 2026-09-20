"""
Publishes alarm changes to alarms/changes over one persistent MQTT connection.

Changes queue while the broker is away and go out in order once it is back. The queue is
bounded, so a long outage costs the oldest changes, never unbounded memory; a consumer
that needs the full picture reads it from AlarmService.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

from aiomqtt import Client
from cogniboiler_observability import MQTT_PUBLISHED
from cogniboiler_runtime import MqttSession

from alert_manager.payloads import TOPIC_CHANGES, change_payload
from alert_manager.views import AlarmView, TransitionView

logger = logging.getLogger(__name__)

QUEUE_LIMIT: int = 1000
RECONNECT_DELAY_S: float = 5.0


class AlarmChangePublisher:
    """Queue-backed publisher of alarm changes; a ChangeListener for the processor."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_id: str = "alert-manager-publisher",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._client_id = client_id
        self._queue: deque[bytes] = deque()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="Alarm change publisher",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )

    def alarm_changed(self, alarm: AlarmView, transition: TransitionView) -> None:
        if len(self._queue) >= QUEUE_LIMIT:
            self._queue.popleft()
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.error(
                    "Alarm change queue full: %d changes dropped so far", self._dropped
                )
        self._queue.append(change_payload(alarm, transition))
        self._wakeup.set()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="alarm-change-publisher")

    async def aclose(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            identifier=self._client_id,
            username=self._username,
            password=self._password,
        )

    async def _run(self) -> None:
        await self._session.run(self._publish_queued)

    async def _publish_queued(self, client: Client) -> None:
        """Send what is queued, then wait for the next change, for as long as the
        session holds. A publish that fails ends the session, and the queue keeps what
        it had: the change goes out on the next connection, in order."""
        while True:
            while self._queue:
                await client.publish(TOPIC_CHANGES, self._queue[0], qos=1)
                MQTT_PUBLISHED.labels(TOPIC_CHANGES).inc()
                self._queue.popleft()
            self._wakeup.clear()
            if not self._queue:
                await self._wakeup.wait()
