"""Alert-manager entry point: arguments, then alert_manager.service runs until stopped."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging
from cogniboiler_runtime import run_service

from alert_manager.grpc_server import DEFAULT_PORT
from alert_manager.service import main

DEFAULT_METRICS_PORT = 9104


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Alert Manager")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--grpc-port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--liveness-file",
        type=Path,
        default=None,
        help="refresh this file while connected to the broker (container healthcheck)",
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


def run(argv: Sequence[str] | None = None) -> int:
    configure_logging("alert-manager")
    args = parse_args(argv)
    return run_service(lambda: main(args))


if __name__ == "__main__":
    sys.exit(run())
