"""CLI entry point for the PLC gRPC server."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging
from cogniboiler_runtime import run_service

from plc_controller.client import DEFAULT_PHYSICS_TARGET
from plc_controller.events import DEFAULT_MQTT_HOST, DEFAULT_MQTT_PORT
from plc_controller.server import DEFAULT_HOST, DEFAULT_PORT, serve

DEFAULT_METRICS_PORT = 9102


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler PLC Controller")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="interface for the gRPC server; 0.0.0.0 only inside a private network",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="gRPC listen port"
    )
    parser.add_argument(
        "--physics-target",
        default=DEFAULT_PHYSICS_TARGET,
        help="PhysicsService host:port target",
    )
    parser.add_argument(
        "--mqtt-host", default=DEFAULT_MQTT_HOST, help="MQTT broker host"
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=DEFAULT_MQTT_PORT, help="MQTT broker port"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=DEFAULT_METRICS_PORT,
        help="Prometheus /metrics port, 0 to disable",
    )
    parser.add_argument(
        "--metrics-host", default="127.0.0.1", help="interface for /metrics"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging("plc-controller")
    args = parse_args(argv)
    return run_service(
        lambda: serve(
            host=args.host,
            port=args.port,
            physics_target=args.physics_target,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_username=os.environ.get("MQTT_USERNAME", "plc-controller"),
            mqtt_password=os.environ.get("MQTT_PASSWORD") or None,
            metrics_port=args.metrics_port,
            metrics_host=args.metrics_host,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
