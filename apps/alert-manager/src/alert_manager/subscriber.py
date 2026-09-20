"""
MQTT subscriber that feeds alarm conditions and snapshots to the alarm processor.

The broker session is persistent (a fixed client id, clean_session off), so condition
messages published with QoS 1 while the alert manager restarts are delivered when it is
back. A database hiccup is retried a few times before a message is given up.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from aiomqtt import Client
from cogniboiler_observability import MQTT_RECEIVED
from cogniboiler_runtime import MqttSession
from sqlalchemy.exc import SQLAlchemyError

from alert_manager.metrics import MESSAGES_FAILED
from alert_manager.payloads import (
    SUBSCRIBE_TOPIC,
    TOPIC_SNAPSHOT,
    ConditionReport,
    PayloadError,
    SnapshotReport,
    parse_condition,
    parse_snapshot,
)

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S: float = 5.0
STORE_ATTEMPTS: int = 3
STORE_RETRY_DELAY_S: float = 1.0


class MessageHandler(Protocol):
    async def handle_condition(self, report: ConditionReport) -> None: ...

    async def handle_snapshot(self, report: SnapshotReport) -> None: ...


class AlertSubscriber:
    """Consume alarm messages from MQTT and hand them to the processor."""

    def __init__(
        self,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        handler: MessageHandler | None = None,
        *,
        client_id: str = "alert-manager",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = mqtt_host
        self._port = mqtt_port
        self._username = username
        self._password = password
        self._handler = handler
        self._client_id = client_id
        self._received = 0
        self._processed = 0
        self._skipped = 0
        self._failed = 0
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="AlertManager",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )

    @property
    def connected(self) -> bool:
        """True while subscribed to the broker; the liveness file follows it."""
        return self._session.connected

    @property
    def stats(self) -> dict[str, int]:
        return {
            "received": self._received,
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
        }

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        self._received += 1
        MQTT_RECEIVED.labels(topic).inc()
        handler = self._handler
        if handler is None:
            self._skipped += 1
            return
        try:
            if topic == TOPIC_SNAPSHOT:
                snapshot = parse_snapshot(raw_payload)
                await self._with_retries(
                    topic, lambda: handler.handle_snapshot(snapshot)
                )
            else:
                report = parse_condition(topic, raw_payload)
                await self._with_retries(
                    topic, lambda: handler.handle_condition(report)
                )
        except PayloadError as exc:
            self._skipped += 1
            logger.warning("Alarm message on %s rejected: %s", topic, exc)
            return
        except Exception:
            self._failed += 1
            MESSAGES_FAILED.inc()
            logger.exception("Alarm message on %s could not be processed", topic)
            return
        self._processed += 1

    @staticmethod
    async def _with_retries(
        topic: str, operation: Callable[[], Awaitable[None]]
    ) -> None:
        for attempt in range(1, STORE_ATTEMPTS + 1):
            try:
                await operation()
                return
            except SQLAlchemyError as exc:
                if attempt == STORE_ATTEMPTS:
                    raise
                logger.warning(
                    "Storing alarm message from %s failed (attempt %d/%d): %s",
                    topic,
                    attempt,
                    STORE_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(STORE_RETRY_DELAY_S)

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            identifier=self._client_id,
            username=self._username,
            password=self._password,
            clean_session=False,
        )

    async def _consume(self, client: Client) -> None:
        """One session: subscribe, then hand every payload to the processor.

        The session is persistent and the subscription is QoS 1, so conditions published
        while the alert manager was away are delivered once it is back.
        """
        await client.subscribe(SUBSCRIBE_TOPIC, qos=1)
        logger.info(
            "AlertManager subscribed to %s on %s:%d",
            SUBSCRIBE_TOPIC,
            self._host,
            self._port,
        )
        async for message in client.messages:
            payload = message.payload
            if not isinstance(payload, bytes | bytearray):
                self._received += 1
                self._skipped += 1
                continue
            await self._handle_message(str(message.topic), bytes(payload))

    async def run(self) -> None:
        """Consume alarm messages for as long as the service lives."""
        await self._session.run(self._consume)
