"""CLI entry point for the PLC gRPC server."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging

from plc_controller.server import DEFAULT_PORT, serve

DEFAULT_METRICS_PORT = 9102


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
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=DEFAULT_METRICS_PORT,
        help="Prometheus /metrics port, 0 to disable",
    )
    parser.add_argument(
        "--metrics-host", default="127.0.0.1", help="interface for /metrics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging("plc-controller")
    args = parse_args()
    asyncio.run(
        serve(
            port=args.port,
            physics_target=args.physics_target,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_username=os.environ.get("MQTT_USERNAME", "plc-controller"),
            mqtt_password=os.environ.get("MQTT_PASSWORD") or None,
            metrics_port=args.metrics_port,
            metrics_host=args.metrics_host,
        ),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
