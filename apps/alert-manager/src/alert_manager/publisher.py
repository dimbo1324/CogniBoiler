"""
Publishes alarm changes to alarms/changes over one persistent MQTT connection.

Changes queue while the broker is away and go out in order once it is back. The queue is
bounded, so a long outage costs the oldest changes, never unbounded memory; a consumer
that needs the full picture reads it from AlarmService. Closing gives a connected
publisher a bounded moment to send what is still queued: those changes are already
committed, and the historian has no other way to learn them.
"""

from __future__ import annotations

import logging

from aiomqtt import Client
from cogniboiler_observability import MQTT_PUBLISH_ERRORS, MQTT_PUBLISHED
from cogniboiler_runtime import (
    DEFAULT_QUEUE_LIMIT,
    DEFAULT_RECONNECT_DELAY_S,
    MqttSession,
    QueuedMessage,
    QueuedMqttPublisher,
)
from cogniboiler_runtime.topics import TOPIC_ALARM_CHANGES

from alert_manager.metrics import CHANGES_DROPPED
from alert_manager.payloads import change_payload
from alert_manager.views import AlarmView, TransitionView

logger = logging.getLogger(__name__)

QUEUE_LIMIT: int = DEFAULT_QUEUE_LIMIT
RECONNECT_DELAY_S: float = DEFAULT_RECONNECT_DELAY_S
CLOSE_DRAIN_TIMEOUT_S: float = 2.0


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
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="Alarm change publisher",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )
        self._queue: QueuedMqttPublisher[Client] = QueuedMqttPublisher(
            self._session,
            name="Alarm change publisher",
            limit=QUEUE_LIMIT,
            on_published=lambda topic: MQTT_PUBLISHED.labels(topic).inc(),
            on_publish_error=lambda topic: MQTT_PUBLISH_ERRORS.labels(topic).inc(),
            logger=logger,
        )

    def alarm_changed(self, alarm: AlarmView, transition: TransitionView) -> None:
        dropped = self._queue.dropped
        self._queue.enqueue(
            QueuedMessage(TOPIC_ALARM_CHANGES, change_payload(alarm, transition))
        )
        CHANGES_DROPPED.inc(self._queue.dropped - dropped)

    @property
    def connected(self) -> bool:
        return self._session.connected

    @property
    def dropped(self) -> int:
        """Alarm changes lost to a full queue."""
        return self._queue.dropped

    def start(self) -> None:
        self._queue.start()

    async def aclose(self) -> None:
        """Send what is queued while the broker is there, within a bound; then stop."""
        await self._queue.aclose(CLOSE_DRAIN_TIMEOUT_S)
        if self._queue.pending:
            logger.warning(
                "Alarm change publisher closed with %d alarm changes unpublished",
                self._queue.pending,
            )

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            identifier=self._client_id,
            username=self._username,
            password=self._password,
        )
