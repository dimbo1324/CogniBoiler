"""CLI entry point for the PLC gRPC server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from plc_controller.server import DEFAULT_PORT, serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler PLC Controller")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="gRPC listen port"
    )
    parser.add_argument(
        "--physics-target",
        default="localhost:50052",
        help="PhysicsService host:port target",
    )
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    asyncio.run(
        serve(
            port=args.port,
            physics_target=args.physics_target,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
        ),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
