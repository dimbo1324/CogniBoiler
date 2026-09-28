"""
OPC UA Server entry point: arguments and the service runner; opcua_server.service runs it.

Usage:
    uv run --package opcua-server python -m opcua_server --mqtt-host localhost --opc-port 4840
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging
from cogniboiler_runtime import run_service

from opcua_server.server import DEFAULT_PORT
from opcua_server.service import main

DEFAULT_METRICS_PORT = 9105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler OPC UA Server")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--opc-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--plc-target", default="localhost:50051")
    parser.add_argument("--alarm-target", default="localhost:50053")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--max-update-hz", type=float, default=5.0)
    parser.add_argument("--plc-interval-s", type=float, default=1.0)
    parser.add_argument("--alarm-interval-s", type=float, default=5.0)
    parser.add_argument("--metrics-port", type=int, default=DEFAULT_METRICS_PORT)
    parser.add_argument("--metrics-host", default="127.0.0.1")
    return parser.parse_args()


def run() -> int:
    configure_logging("opcua-server")
    args = parse_args()
    return run_service(lambda: main(args))


if __name__ == "__main__":
    sys.exit(run())
