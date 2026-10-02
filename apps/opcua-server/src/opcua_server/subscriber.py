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

A message that is empty, not bytes, not the expected protobuf, or has no timestamp is
skipped and counted: proto3 parses zero bytes as a message with every field at zero,
which would otherwise be published as a plant at 0 Pa with Good quality. A defect while
projecting a message is logged as such and the message skipped; it never reaches the
MQTT session, which would take it for a broker outage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import cogniboiler_pb2 as pb
from aiomqtt import Client
from cogniboiler_observability import MQTT_RECEIVED
from cogniboiler_runtime import DEFAULT_RECONNECT_DELAY_S, MqttSession, consume
from cogniboiler_runtime.topics import (
    FILTER_SENSORS,
    TOPIC_ALARM_CHANGES,
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_PLANT,
    TOPIC_TURBINE,
)
from google.protobuf.message import DecodeError

from opcua_server.address_space import BOILER_FIELD_TO_NODEID, TURBINE_FIELD_TO_NODEID
from opcua_server.metrics import BRIDGE_SKIPPED
from opcua_server.projection import (
    Update,
    field_updates,
    plant_updates,
    sensor_qualities,
)
from opcua_server.server import QUALITY_GOOD, CogniBoilerOPCServer

logger = logging.getLogger(__name__)

SUBSCRIPTIONS: tuple[tuple[str, int], ...] = (
    (FILTER_SENSORS, 0),
    (TOPIC_ALARM_CHANGES, 1),
)
RECONNECT_DELAY_S: float = DEFAULT_RECONNECT_DELAY_S
MIN_FLUSH_INTERVAL_S: float = 0.05

SKIP_UNKNOWN_TOPIC = "unknown_topic"
SKIP_EMPTY = "empty"
SKIP_MALFORMED = "malformed"
SKIP_DEFECT = "defect"

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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._opc = opc_server
        self._host = mqtt_host
        self._port = mqtt_port
        self._username = mqtt_username
        self._password = mqtt_password
        self._interval_s = 1.0 / max_update_hz if max_update_hz > 0 else 0.0
        self._alarms_changed = alarms_changed
        self._clock = clock
        self._qualities: dict[int, int] = {}
        self._last_applied: dict[str, float] = {}
        self._pending: dict[str, _Parsed] = {}
        self._messages_received: int = 0
        self._messages_mapped: int = 0
        self._messages_skipped: int = 0
        self._reported_defects: set[tuple[str, type[BaseException]]] = set()
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
        _require_timestamp(msg.timestamp_ms)
        self._qualities = sensor_qualities(msg)
        return plant_updates(msg), msg.timestamp_ms

    def _parse_boiler(self, raw: bytes) -> _Parsed:
        msg = pb.BoilerStateMsg()
        msg.ParseFromString(raw)
        _require_timestamp(msg.timestamp_ms)
        return field_updates(msg, BOILER_FIELD_TO_NODEID), msg.timestamp_ms

    def _parse_turbine(self, raw: bytes) -> _Parsed:
        msg = pb.TurbineStateMsg()
        msg.ParseFromString(raw)
        _require_timestamp(msg.timestamp_ms)
        return field_updates(msg, TURBINE_FIELD_TO_NODEID), msg.timestamp_ms

    # ─── Message handling ─────────────────────────────────────────────────────

    async def _handle_message(self, topic: str, raw_payload: object) -> None:
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
            self._skip(SKIP_UNKNOWN_TOPIC)
            if topic != TOPIC_HEARTBEAT:
                logger.debug("No handler for topic: %s", topic)
            return
        if not isinstance(raw_payload, bytes | bytearray) or not raw_payload:
            self._skip(SKIP_EMPTY)
            logger.debug("Empty or non-binary payload on %s skipped", topic)
            return

        try:
            parsed = parser(bytes(raw_payload))
        except (DecodeError, ValueError) as exc:
            self._skip(SKIP_MALFORMED)
            logger.warning("Protobuf decode error on %s: %s", topic, exc)
            return
        except Exception as exc:
            self._defect(topic, exc)
            return

        now = self._clock()
        if now - self._last_applied.get(topic, float("-inf")) < self._interval_s:
            self._pending[topic] = parsed
            return
        await self._apply(topic, parsed, now)

    async def _apply(self, topic: str, parsed: _Parsed, now: float) -> None:
        updates, timestamp_ms = parsed
        self._last_applied[topic] = now
        self._pending.pop(topic, None)
        try:
            for node_id, value in updates:
                try:
                    await self._opc.update_variable(
                        node_id,
                        value,
                        quality=self._qualities.get(node_id, QUALITY_GOOD),
                        source_timestamp_ms=timestamp_ms,
                    )
                except KeyError:
                    logger.warning("OPC UA node %d not found", node_id)
        except Exception as exc:
            self._defect(topic, exc)
            return
        self._messages_mapped += 1

    async def flush_pending(self) -> None:
        """Apply held-back messages whose turn has come."""
        interval = max(self._interval_s, MIN_FLUSH_INTERVAL_S)
        while True:
            await asyncio.sleep(interval)
            now = self._clock()
            for topic, parsed in list(self._pending.items()):
                if (
                    now - self._last_applied.get(topic, float("-inf"))
                    >= self._interval_s
                ):
                    await self._apply(topic, parsed, now)

    def _skip(self, reason: str) -> None:
        self._messages_skipped += 1
        BRIDGE_SKIPPED.labels(reason).inc()

    def _defect(self, topic: str, exc: Exception) -> None:
        """A bug, not bad input: say so once per topic and kind, then keep going."""
        self._skip(SKIP_DEFECT)
        key = (topic, type(exc))
        if key in self._reported_defects:
            logger.debug("OPC UA bridge failed again on %s: %r", topic, exc)
            return
        self._reported_defects.add(key)
        logger.error("OPC UA bridge failed on %s; message skipped", topic, exc_info=exc)

    def _open_client(self) -> Client:
        return Client(
            hostname=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
        )

    async def _consume(self, client: Client) -> None:
        await consume(client, SUBSCRIPTIONS, self._handle_message)

    async def run(self) -> None:
        """Project what the plant publishes onto the address space, session after session."""
        await self._session.run(self._consume)


def _require_timestamp(timestamp_ms: int) -> None:
    if timestamp_ms <= 0:
        raise ValueError("message without a timestamp")
