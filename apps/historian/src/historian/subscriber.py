"""
MQTT -> InfluxDB historian subscriber.

Flow:
    MQTT broker
        -> HistorianSubscriber.run()
        -> _handle_message(topic, raw_payload)
        -> point builders (historian.writer, historian.points)
        -> batched InfluxWriter writes in a worker thread

Topic contract:
    sensors/plant            ← PlantStatusMsg  (protobuf): plant status, KPIs, labels
    sensors/boiler           ← BoilerStateMsg  (protobuf)
    sensors/turbine          ← TurbineStateMsg (protobuf)
    sensors/system/heartbeat ← UTF-8 timestamp (skipped)
    alarms/changes           ← JSON alarm after a state change (alert-manager)
    plc/events               ← JSON PLC event (plc-controller)
    status/<service>         ← retained online / offline

Boiler and turbine values are tagged with the scenario of the latest plant status;
scenario loads and fault changes are written as simulation events. The session is
persistent, so alarm changes and PLC events published while the historian restarts are
delivered once it is back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import cogniboiler_pb2 as pb
from aiomqtt import Client
from google.protobuf.message import DecodeError

from historian.points import (
    RunLabels,
    build_alarm_change_point,
    build_availability_point,
    build_plant_point,
    build_plc_event_point,
    build_simulation_event_point,
)
from historian.writer import (
    InfluxWriter,
    PointLike,
    build_boiler_point,
    build_turbine_point,
)

logger = logging.getLogger(__name__)

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_PLANT: str = "sensors/plant"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"
TOPIC_ALARM_CHANGES: str = "alarms/changes"
TOPIC_PLC_EVENTS: str = "plc/events"
TOPIC_STATUS_PREFIX: str = "status/"

SUBSCRIPTIONS: tuple[tuple[str, int], ...] = (
    ("sensors/#", 0),
    (TOPIC_ALARM_CHANGES, 1),
    (TOPIC_PLC_EVENTS, 1),
    ("status/+", 1),
)
SUBSCRIBE_TOPIC: str = "sensors/#"
RECONNECT_DELAY_S: float = 5.0


class HistorianSubscriber:
    """
    Async MQTT subscriber that persists telemetry and events to InfluxDB.

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
        *,
        batch_size: int = 1,
        flush_interval_s: float = 2.0,
        client_id: str | None = None,
    ) -> None:
        self._writer = writer
        self._host = mqtt_host
        self._port = mqtt_port
        self._client_id = client_id
        self._batch_size = max(batch_size, 1)
        self._buffer: list[PointLike] = []
        self._flush_interval_s = max(flush_interval_s, 0.1)
        self._last_flush_at = time.monotonic()
        self._labels = RunLabels()
        self._received: int = 0
        self._stored: int = 0
        self._skipped: int = 0
        self._connected = False
        self._handlers: dict[str, Callable[[bytes], list[PointLike] | None]] = {
            TOPIC_PLANT: self._plant,
            TOPIC_BOILER: self._boiler,
            TOPIC_TURBINE: self._turbine,
            TOPIC_ALARM_CHANGES: self._alarm_change,
            TOPIC_PLC_EVENTS: self._plc_event,
        }

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

    # ─── Message handling ─────────────────────────────────────────────────────

    def _plant(self, raw: bytes) -> list[PointLike] | None:
        msg = pb.PlantStatusMsg()
        msg.ParseFromString(raw)
        events = self._labels.update(msg)
        return [build_plant_point(msg)] + [
            build_simulation_event_point(event) for event in events
        ]

    def _boiler(self, raw: bytes) -> list[PointLike] | None:
        msg = pb.BoilerStateMsg()
        msg.ParseFromString(raw)
        return [build_boiler_point(msg, self._labels.scenario)]

    def _turbine(self, raw: bytes) -> list[PointLike] | None:
        msg = pb.TurbineStateMsg()
        msg.ParseFromString(raw)
        return [build_turbine_point(msg, self._labels.scenario)]

    @staticmethod
    def _json(raw: bytes) -> dict[str, Any] | None:
        try:
            value = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError, json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _alarm_change(self, raw: bytes) -> list[PointLike] | None:
        payload = self._json(raw)
        point = build_alarm_change_point(payload) if payload is not None else None
        return [point] if point is not None else None

    def _plc_event(self, raw: bytes) -> list[PointLike] | None:
        payload = self._json(raw)
        point = build_plc_event_point(payload) if payload is not None else None
        return [point] if point is not None else None

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        """
        Turn one MQTT message into points and store them.

        Skips the heartbeat, unknown topics and malformed payloads.
        """
        self._received += 1

        if topic.startswith(TOPIC_STATUS_PREFIX):
            point = build_availability_point(
                topic.removeprefix(TOPIC_STATUS_PREFIX), raw_payload
            )
            points = [point] if point is not None else None
        elif (handler := self._handlers.get(topic)) is not None:
            try:
                points = handler(raw_payload)
            except (DecodeError, ValueError) as exc:
                logger.warning("Malformed payload on %s: %s", topic, exc)
                points = None
        else:
            if topic != TOPIC_HEARTBEAT:
                logger.debug("No handler for topic: %s", topic)
            points = None

        if not points:
            self._skipped += 1
            return
        for point in points:
            await self._store_point(point)

    async def _store_point(self, point: PointLike) -> None:
        """Append to the current batch and flush when needed."""
        self._buffer.append(point)
        if len(self._buffer) >= self._batch_size:
            await self._flush()
            return

        if time.monotonic() - self._last_flush_at >= self._flush_interval_s:
            await self._flush()
            return

    async def _flush(self) -> None:
        """Flush the current batch to InfluxDB without blocking the event loop."""
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush_at = time.monotonic()
        if len(batch) == 1:
            await asyncio.to_thread(self._writer.write_point, batch[0])
        else:
            await asyncio.to_thread(self._writer.write_points, batch)
        self._stored += len(batch)

    async def flush_periodically(self) -> None:
        """Flush a partial batch when messages stop, e.g. while the plant is paused."""
        while True:
            await asyncio.sleep(self._flush_interval_s)
            if time.monotonic() - self._last_flush_at >= self._flush_interval_s:
                await self._flush()

    async def run(self) -> None:
        while True:
            try:
                async with Client(
                    hostname=self._host,
                    port=self._port,
                    identifier=self._client_id,
                    clean_session=False if self._client_id else None,
                ) as client:
                    logger.info(
                        "Historian connected to MQTT %s:%d",
                        self._host,
                        self._port,
                    )
                    for topic, qos in SUBSCRIPTIONS:
                        await client.subscribe(topic, qos=qos)
                    self._connected = True
                    try:
                        async for message in client.messages:
                            payload = message.payload
                            await self._handle_message(
                                str(message.topic),
                                payload if isinstance(payload, bytes) else b"",
                            )
                    finally:
                        self._connected = False
            except Exception as exc:
                logger.warning(
                    "Historian MQTT error: %s — retrying in %.0fs",
                    exc,
                    RECONNECT_DELAY_S,
                )
                await self._flush()
                await asyncio.sleep(RECONNECT_DELAY_S)
