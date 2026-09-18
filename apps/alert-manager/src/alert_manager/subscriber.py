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
    ) -> None:
        self._host = mqtt_host
        self._port = mqtt_port
        self._handler = handler
        self._client_id = client_id
        self._received = 0
        self._processed = 0
        self._skipped = 0
        self._failed = 0
        self._connected = False
        self._failing = False

    @property
    def connected(self) -> bool:
        """True while subscribed to the broker; the liveness file follows it."""
        return self._connected

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

    async def run(self) -> None:
        """Run forever, reconnecting after broker failures."""
        while True:
            try:
                async with Client(
                    hostname=self._host,
                    port=self._port,
                    identifier=self._client_id,
                    clean_session=False,
                ) as client:
                    await client.subscribe(SUBSCRIBE_TOPIC, qos=1)
                    self._connected = True
                    logger.info(
                        "AlertManager subscribed to %s on %s:%d",
                        SUBSCRIBE_TOPIC,
                        self._host,
                        self._port,
                    )
                    self._failing = False
                    try:
                        async for message in client.messages:
                            payload = message.payload
                            if not isinstance(payload, bytes | bytearray):
                                self._received += 1
                                self._skipped += 1
                                continue
                            await self._handle_message(
                                str(message.topic), bytes(payload)
                            )
                    finally:
                        self._connected = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._failing:
                    logger.warning(
                        "AlertManager MQTT error: %s — retrying every %.0fs",
                        exc,
                        RECONNECT_DELAY_S,
                    )
                self._failing = True
                await asyncio.sleep(RECONNECT_DELAY_S)
