"""
OPC UA Server entry point.

Starts the CogniBoiler OPC UA server and bridges MQTT sensor
topics to OPC UA variable nodes in real time.

Usage:
    uv run --package opcua-server python -m opcua_server
    uv run --package opcua-server python -m opcua_server --mqtt-host localhost --opc-port 4840
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from opcua_server.server import CogniBoilerOPCServer
from opcua_server.subscriber import MQTTOPCBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("opcua_server")


async def main(mqtt_host: str, mqtt_port: int, opc_port: int) -> None:
    opc_server = CogniBoilerOPCServer()
    bridge = MQTTOPCBridge(
        opc_server=opc_server,
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
    )

    logger.info(
        "Starting OPC UA Server: opc=opc.tcp://0.0.0.0:%d  mqtt=%s:%d",
        opc_port,
        mqtt_host,
        mqtt_port,
    )

    await opc_server.start()
    logger.info(
        "OPC UA server running — connect with UaExpert at opc.tcp://localhost:%d",
        opc_port,
    )

    async def log_stats() -> None:
        while True:
            await asyncio.sleep(30)
            stats = bridge.stats
            logger.info(
                "Bridge stats: received=%d mapped=%d skipped=%d",
                stats["received"],
                stats["mapped"],
                stats["skipped"],
            )

    try:
        await asyncio.gather(
            bridge.run(),
            log_stats(),
        )
    finally:
        await opc_server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler OPC UA Server")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--opc-port", type=int, default=4840)
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    args = parse_args()
    asyncio.run(
        main(
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            opc_port=args.opc_port,
        )
    )
