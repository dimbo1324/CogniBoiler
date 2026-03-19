"""
MQTT -> InfluxDB historian subscriber.

Subscribes to sensors/boiler and sensors/turbine, deserializes
protobuf payloads, builds InfluxDB Points, and writes via InfluxWriter.

Flow:
    MQTT broker [sensors/#]
        -> HistorianSubscriber.run()
        -> _handle_message(topic, raw_payload)
        -> build_boiler_point(msg) or build_turbine_point(msg)
        -> InfluxWriter.write_point(point)

Topic contract:
    sensors/boiler          ← BoilerStateMsg  (protobuf)
    sensors/turbine         ← TurbineStateMsg (protobuf)
    sensors/system/heartbeat ← UTF-8 timestamp (skipped)
"""

from __future__ import annotations

import asyncio
import logging

import cogniboiler_pb2 as pb
from aiomqtt import Client

from historian.writer import InfluxWriter, build_boiler_point, build_turbine_point

logger = logging.getLogger(__name__)

SUBSCRIBE_TOPIC: str = "sensors/#"

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"


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
        Deserialize one MQTT protobuf message and write to InfluxDB.

        Skips:
          - Heartbeat (sensors/system/heartbeat)
          - Unknown topics
          - Malformed protobuf payloads
        """
        self._received += 1

        if topic == TOPIC_HEARTBEAT:
            self._skipped += 1
            return

        if topic == TOPIC_BOILER:
            try:
                msg = pb.BoilerStateMsg()
                msg.ParseFromString(raw_payload)
            except Exception as exc:
                self._skipped += 1
                logger.warning("Protobuf decode error on %s: %s", topic, exc)
                return
            point = build_boiler_point(msg)
            self._writer.write_point(point)
            self._stored += 1
            return

        if topic == TOPIC_TURBINE:
            try:
                msg = pb.TurbineStateMsg()
                msg.ParseFromString(raw_payload)
            except Exception as exc:
                self._skipped += 1
                logger.warning("Protobuf decode error on %s: %s", topic, exc)
                return
            point = build_turbine_point(msg)
            self._writer.write_point(point)
            self._stored += 1
            return

        # Unknown topic — not part of our schema
        self._skipped += 1
        logger.debug("No handler for topic: %s", topic)

    async def run(self) -> None:
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
                    await client.subscribe(SUBSCRIBE_TOPIC)
                    async for message in client.messages:
                        await self._handle_message(
                            str(message.topic),
                            message.payload,
                        )
            except Exception as exc:
                logger.warning("Historian MQTT error: %s — retrying in 5s", exc)
                await asyncio.sleep(5.0)
