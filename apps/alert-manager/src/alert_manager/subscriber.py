"""MQTT subscriber that persists PLC alarm events."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiomqtt import Client
from sqlalchemy import select

from alert_manager.db import AsyncSessionLocal
from alert_manager.models import AlarmEvent

logger = logging.getLogger(__name__)

SUBSCRIBE_TOPIC: str = "alerts/#"


class AlertSubscriber:
    """Consume alarm events from MQTT and store them in PostgreSQL."""

    def __init__(self, mqtt_host: str = "localhost", mqtt_port: int = 1883) -> None:
        self._host = mqtt_host
        self._port = mqtt_port
        self._received = 0
        self._stored = 0
        self._skipped = 0
        self._connected = False

    @property
    def connected(self) -> bool:
        """True while subscribed to the broker; the liveness file follows it."""
        return self._connected

    @property
    def stats(self) -> dict[str, int]:
        return {
            "received": self._received,
            "stored": self._stored,
            "skipped": self._skipped,
        }

    async def _persist_alarm(self, topic: str, payload: dict[str, Any]) -> None:
        alarm_id = str(payload["alarm_id"])

        async with AsyncSessionLocal() as session:
            existing = await session.scalar(
                select(AlarmEvent).where(AlarmEvent.alarm_id == alarm_id)
            )
            if existing is not None:
                self._skipped += 1
                return

            session.add(
                AlarmEvent(
                    alarm_id=alarm_id,
                    source_service=str(payload["source_service"]),
                    severity=str(payload["severity"]),
                    parameter=str(payload["parameter"]),
                    value=float(payload["value"]),
                    threshold=float(payload["threshold"]),
                    action=str(payload["action"]),
                    message=str(payload["message"]),
                    topic=topic,
                    occurred_at_ms=int(payload["timestamp_ms"]),
                    acknowledged=False,
                    acknowledged_at_ms=None,
                    cleared=False,
                )
            )
            await session.commit()
            self._stored += 1

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        self._received += 1
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except Exception as exc:
            self._skipped += 1
            logger.warning("Alarm decode error on %s: %s", topic, exc)
            return

        required = {
            "alarm_id",
            "source_service",
            "severity",
            "parameter",
            "value",
            "threshold",
            "action",
            "message",
            "timestamp_ms",
        }
        if not required <= set(payload):
            self._skipped += 1
            logger.warning("Alarm payload missing required fields on %s", topic)
            return

        await self._persist_alarm(topic, payload)

    async def run(self) -> None:
        """Run forever, reconnecting on MQTT errors."""
        while True:
            try:
                async with Client(hostname=self._host, port=self._port) as client:
                    logger.info(
                        "AlertManager connected to MQTT %s:%d",
                        self._host,
                        self._port,
                    )
                    await client.subscribe(SUBSCRIBE_TOPIC)
                    self._connected = True
                    try:
                        async for message in client.messages:
                            await self._handle_message(
                                str(message.topic),
                                message.payload,
                            )
                    finally:
                        self._connected = False
            except Exception as exc:
                logger.warning("AlertManager MQTT error: %s — retrying in 5s", exc)
                await asyncio.sleep(5.0)
