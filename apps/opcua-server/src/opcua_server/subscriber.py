"""
MQTT -> OPC UA bridge subscriber (protobuf edition).

Subscribes to sensors/boiler and sensors/turbine, deserializes
protobuf payloads, and updates OPC UA variable nodes field-by-field.

Flow:
    MQTT broker [sensors/#]
        -> MQTTOPCBridge.run()
        -> _handle_message(topic, raw_payload)
        -> ParseFromString(raw_payload) -> BoilerStateMsg | TurbineStateMsg
        -> for each field: CogniBoilerOPCServer.update_variable(node_id, value)

Topic contract:
    sensors/boiler           ← BoilerStateMsg  (protobuf)
    sensors/turbine          ← TurbineStateMsg (protobuf)
    sensors/system/heartbeat ← UTF-8 timestamp (skipped)
"""

from __future__ import annotations

import logging

import cogniboiler_pb2 as pb
from asyncio_mqtt import Client, MqttError

from opcua_server.address_space import BOILER_FIELD_TO_NODEID, TURBINE_FIELD_TO_NODEID
from opcua_server.server import CogniBoilerOPCServer

logger = logging.getLogger(__name__)

SUBSCRIBE_TOPIC: str = "sensors/#"

TOPIC_BOILER: str = "sensors/boiler"
TOPIC_TURBINE: str = "sensors/turbine"
TOPIC_HEARTBEAT: str = "sensors/system/heartbeat"


class MQTTOPCBridge:
    """
    Bridges MQTT sensor topics to OPC UA variable nodes.

    Usage:
        bridge = MQTTOPCBridge(opc_server, mqtt_host="localhost")
        await bridge.run()   # blocks, reconnects on disconnect
    """

    def __init__(
        self,
        opc_server: CogniBoilerOPCServer,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
    ) -> None:
        self._opc = opc_server
        self._host = mqtt_host
        self._port = mqtt_port
        self._messages_received: int = 0
        self._messages_mapped: int = 0
        self._messages_skipped: int = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "received": self._messages_received,
            "mapped": self._messages_mapped,
            "skipped": self._messages_skipped,
        }

    async def _handle_message(self, topic: str, raw_payload: bytes) -> None:
        """
        Deserialize one MQTT protobuf message and update OPC UA nodes.

        Skips:
          - Heartbeat topic (sensors/system/heartbeat)
          - Unknown topics
          - Malformed protobuf payloads
        """
        self._messages_received += 1

        if topic == TOPIC_HEARTBEAT:
            self._messages_skipped += 1
            return

        if topic == TOPIC_BOILER:
            await self._handle_boiler(raw_payload)
            return

        if topic == TOPIC_TURBINE:
            await self._handle_turbine(raw_payload)
            return

        # Unknown topic — not part of our schema
        self._messages_skipped += 1
        logger.debug("No handler for topic: %s", topic)

    async def _handle_boiler(self, raw_payload: bytes) -> None:
        """Deserialize BoilerStateMsg and update all boiler OPC UA nodes."""
        try:
            msg = pb.BoilerStateMsg()
            msg.ParseFromString(raw_payload)
        except Exception as exc:
            self._messages_skipped += 1
            logger.warning("Protobuf decode error on %s: %s", TOPIC_BOILER, exc)
            return

        for field, node_id in BOILER_FIELD_TO_NODEID.items():
            value = float(getattr(msg, field))
            try:
                await self._opc.update_variable(node_id, value)
            except KeyError:
                logger.warning("OPC UA node %d not found (field: %s)", node_id, field)

        self._messages_mapped += 1

    async def _handle_turbine(self, raw_payload: bytes) -> None:
        """Deserialize TurbineStateMsg and update all turbine OPC UA nodes."""
        try:
            msg = pb.TurbineStateMsg()
            msg.ParseFromString(raw_payload)
        except Exception as exc:
            self._messages_skipped += 1
            logger.warning("Protobuf decode error on %s: %s", TOPIC_TURBINE, exc)
            return

        for field, node_id in TURBINE_FIELD_TO_NODEID.items():
            value = float(getattr(msg, field))
            try:
                await self._opc.update_variable(node_id, value)
            except KeyError:
                logger.warning("OPC UA node %d not found (field: %s)", node_id, field)

        self._messages_mapped += 1

    async def run(self) -> None:
        """
        Subscribe to sensors/# and forward to OPC UA indefinitely.
        Reconnects automatically on broker disconnect.
        """
        while True:
            try:
                async with Client(
                    hostname=self._host,
                    port=self._port,
                ) as client:
                    logger.info(
                        "MQTT->OPC bridge connected to %s:%d",
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
                logger.warning("Bridge MQTT error: %s — retrying in 5s", exc)
