"""
MQTT → InfluxDB historian subscriber.

Subscribes to sensors/# , parses each JSON payload,
builds an InfluxDB Point, and writes it via InfluxWriter.

Flow:
    MQTT broker [sensors/#]
        → HistorianSubscriber.run()
        → _handle_message(topic, raw_payload)
        → build_point(topic, parsed_payload)
        → InfluxWriter.write_point(point)
"""

from __future__ import annotations

import json
import logging

from asyncio_mqtt import Client, MqttError

from historian.writer import InfluxWriter, build_point

logger = logging.getLogger(__name__)

SUBSCRIBE_TOPIC: str = "sensors/#"


class HistorianSubscriber:
    """
    Async MQTT subscriber that persists sensor telemetry to InfluxDB.

    Usage:
        writer = InfluxWriter(...)
        sub = HistorianSubscriber(writer, mqtt_host="localhost")
        await sub.run()   # blocks, reconnects on disconnect
    """

    def __init__(
        self,
        writer: InfluxWriter,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
    ) -> None:
        self._writer = writer
        self._host = mqtt_host
        self._port = mqtt_port
        self._received: int = 0
        self._stored: int = 0
        self._skipped: int = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "received": self._received,
            "stored": self._stored,
            "skipped": self._skipped,
        }

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        """
        Parse one MQTT message and write to InfluxDB.

        Skips:
          - Heartbeat (sensors/system/timestamp_ms)
          - Malformed JSON
          - Topics with no measurement mapping
        """
        self._received += 1

        # Skip heartbeat
        if topic.endswith("/timestamp_ms"):
            self._skipped += 1
            return

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            self._skipped += 1
            logger.warning("JSON decode error on %s: %s", topic, exc)
            return

        point = build_point(topic, payload)
        if point is None:
            self._skipped += 1
            return

        self._writer.write_point(point)
        self._stored += 1

    async def run(self) -> None:
        """
        Subscribe to sensors/# and persist to InfluxDB indefinitely.
        Reconnects automatically on broker disconnect.
        """
        while True:
            try:
                async with Client(
                    hostname=self._host,
                    port=self._port,
                ) as client:
                    logger.info(
                        "Historian connected to MQTT %s:%d",
                        self._host,
                        self._port,
                    )
                    async with client.filtered_messages(SUBSCRIBE_TOPIC) as messages:
                        await client.subscribe(SUBSCRIBE_TOPIC)
                        async for message in messages:
                            await self._handle_message(
                                message.topic,
                                message.payload,
                            )
            except MqttError as exc:
                logger.warning("Historian MQTT error: %s — retrying in 5s", exc)
