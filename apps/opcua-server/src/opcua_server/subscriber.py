"""
MQTT -> OPC UA bridge.

Flow:
    MQTT broker
        -> MQTTOPCBridge.run()
        -> _handle_message(topic, raw_payload)
        -> CogniBoilerOPCServer.update_variable(node_id, value, quality=..., ...)

Topic contract:
    sensors/plant            ← PlantStatusMsg: valves, emissions, condenser, KPIs, health,
                               simulation, instrument qualities
    sensors/boiler           ← BoilerStateMsg  (measured)
    sensors/turbine          ← TurbineStateMsg (measured)
    sensors/system/heartbeat ← UTF-8 timestamp (skipped)
    alarms/changes           ← JSON alarm change: the alarm projection refreshes at once

The plant publishes every simulated step, up to fifty times a second at high simulation
speed. Each topic is applied at most max_update_hz times a second; a message arriving
sooner replaces the pending one, which is applied when its turn comes, so clients always
end on the latest values without the server writing every step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import cogniboiler_pb2 as pb
from aiomqtt import Client
from cogniboiler_observability import MQTT_RECEIVED
from cogniboiler_runtime import MqttSession, subscribe_all
from google.protobuf.message import DecodeError

from opcua_server.address_space import BOILER_FIELD_TO_NODEID, TURBINE_FIELD_TO_NODEID
from opcua_server.projection import Update, plant_updates, sensor_qualities
from opcua_server.server import QUALITY_GOOD, CogniBoilerOPCServer

logger = logging.getLogger(__name__)

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_PLANT: str = "sensors/plant"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"
TOPIC_ALARM_CHANGES: str = "alarms/changes"

SUBSCRIPTIONS: tuple[tuple[str, int], ...] = (
    ("sensors/#", 0),
    (TOPIC_ALARM_CHANGES, 1),
)
RECONNECT_DELAY_S: float = 5.0

_Parsed = tuple[list[Update], int]


class MQTTOPCBridge:
    """
    Bridges MQTT topics to OPC UA variable nodes.

    Usage:
        bridge = MQTTOPCBridge(opc_server, mqtt_host="localhost")
        await bridge.run()   # blocks, reconnects on disconnect
    """

    def __init__(
        self,
        opc_server: CogniBoilerOPCServer,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        *,
        max_update_hz: float = 5.0,
        alarms_changed: asyncio.Event | None = None,
        mqtt_username: str | None = None,
        mqtt_password: str | None = None,
    ) -> None:
        self._opc = opc_server
        self._host = mqtt_host
        self._port = mqtt_port
        self._username = mqtt_username
        self._password = mqtt_password
        self._interval_s = 1.0 / max_update_hz if max_update_hz > 0 else 0.0
        self._alarms_changed = alarms_changed
        self._qualities: dict[int, int] = {}
        self._last_applied: dict[str, float] = {}
        self._pending: dict[str, _Parsed] = {}
        self._messages_received: int = 0
        self._messages_mapped: int = 0
        self._messages_skipped: int = 0
        self._parsers: dict[str, Callable[[bytes], _Parsed]] = {
            TOPIC_PLANT: self._parse_plant,
            TOPIC_BOILER: self._parse_boiler,
            TOPIC_TURBINE: self._parse_turbine,
        }
        self._session: MqttSession[Client] = MqttSession(
            self._open_client,
            name="MQTT to OPC UA bridge",
            reconnect_delay_s=RECONNECT_DELAY_S,
            logger=logger,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "received": self._messages_received,
            "mapped": self._messages_mapped,
            "skipped": self._messages_skipped,
        }

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def _parse_plant(self, raw: bytes) -> _Parsed:
        msg = pb.PlantStatusMsg()
        msg.ParseFromString(raw)
        self._qualities = sensor_qualities(msg)
        return plant_updates(msg), msg.timestamp_ms

    def _parse_boiler(self, raw: bytes) -> _Parsed:
        msg = pb.BoilerStateMsg()
        msg.ParseFromString(raw)
        return _fields(msg, BOILER_FIELD_TO_NODEID), msg.timestamp_ms

    def _parse_turbine(self, raw: bytes) -> _Parsed:
        msg = pb.TurbineStateMsg()
        msg.ParseFromString(raw)
        return _fields(msg, TURBINE_FIELD_TO_NODEID), msg.timestamp_ms

    # ─── Message handling ─────────────────────────────────────────────────────

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        """
        Parse one MQTT message and update the OPC UA nodes it feeds.

        Skips the heartbeat, unknown topics and malformed payloads.
        """
        self._messages_received += 1
        MQTT_RECEIVED.labels(topic).inc()

        if topic == TOPIC_ALARM_CHANGES:
            if self._alarms_changed is not None:
                self._alarms_changed.set()
            self._messages_mapped += 1
            return

        parser = self._parsers.get(topic)
        if parser is None:
            self._messages_skipped += 1
            if topic != TOPIC_HEARTBEAT:
                logger.debug("No handler for topic: %s", topic)
            return

        try:
            parsed = parser(raw_payload)
        except (DecodeError, ValueError) as exc:
            self._messages_skipped += 1
            logger.warning("Protobuf decode error on %s: %s", topic, exc)
            return

        now = time.monotonic()
        if now - self._last_applied.get(topic, float("-inf")) < self._interval_s:
            self._pending[topic] = parsed
            return
        await self._apply(topic, parsed, now)

    async def _apply(self, topic: str, parsed: _Parsed, now: float) -> None:
        updates, timestamp_ms = parsed
        self._last_applied[topic] = now
        self._pending.pop(topic, None)
        for node_id, value in updates:
            try:
                await self._opc.update_variable(
                    node_id,
                    value,
                    quality=self._qualities.get(node_id, QUALITY_GOOD),
                    source_timestamp_ms=timestamp_ms or None,
                )
            except KeyError:
                logger.warning("OPC UA node %d not found", node_id)
        self._messages_mapped += 1

    async def flush_pending(self) -> None:
        """Apply held-back messages whose turn has come."""
        interval = max(self._interval_s, 0.05)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for topic, parsed in list(self._pending.items()):
                if (
                    now - self._last_applied.get(topic, float("-inf"))
                    >= self._interval_s
                ):
                    await self._apply(topic, parsed, now)

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
        )

    async def _consume(self, client: Client) -> None:
        await subscribe_all(client, SUBSCRIPTIONS)
        async for message in client.messages:
            payload = message.payload
            await self._handle_message(
                str(message.topic),
                payload if isinstance(payload, bytes) else b"",
            )

    async def run(self) -> None:
        """Project what the plant publishes onto the address space, session after session."""
        await self._session.run(self._consume)


def _fields(message: Any, mapping: dict[str, int]) -> list[Update]:
    return [
        (node_id, float(getattr(message, field))) for field, node_id in mapping.items()
    ]
