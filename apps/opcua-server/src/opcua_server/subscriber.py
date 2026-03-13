"""
MQTT → OPC UA bridge subscriber.

Subscribes to all sensor topics (sensors/#), parses JSON payload,
maps topic to OPC UA node ID, and writes the value to the server.

Flow:
    MQTT broker [sensors/#]
        → MQTTOPCBridge.run()
        → _handle_message(topic, payload)
        → CogniBoilerOPCServer.update_variable(node_id, value)
"""

from __future__ import annotations

import json
import logging

from asyncio_mqtt import Client, MqttError

from opcua_server.address_space import MQTT_TOPIC_TO_NODEID
from opcua_server.server import CogniBoilerOPCServer

logger = logging.getLogger(__name__)

SUBSCRIBE_TOPIC: str = "sensors/#"


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

    async def _handle_message(self, topic: str, payload: bytes) -> None:
        """
        Parse a single MQTT message and update the OPC UA node.

        Skips:
          - Heartbeat topic (sensors/system/timestamp_ms)
          - Topics not in MQTT_TOPIC_TO_NODEID mapping
          - Malformed JSON payloads
        """
        self._messages_received += 1

        # Skip heartbeat
        if topic.endswith("/timestamp_ms"):
            self._messages_skipped += 1
            return

        node_id = MQTT_TOPIC_TO_NODEID.get(topic)
        if node_id is None:
            self._messages_skipped += 1
            logger.debug("No OPC UA mapping for topic: %s", topic)
            return

        try:
            doc = json.loads(payload)
            value = float(doc["value"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self._messages_skipped += 1
            logger.warning("Bad payload on %s: %s", topic, exc)
            return

        try:
            await self._opc.update_variable(node_id, value)
            self._messages_mapped += 1
        except KeyError:
            self._messages_skipped += 1
            logger.warning("OPC UA node %d not found", node_id)

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
                        "MQTT→OPC bridge connected to %s:%d",
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
