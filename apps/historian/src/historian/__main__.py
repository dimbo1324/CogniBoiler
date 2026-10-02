"""
Historian entry point: arguments, then historian.service runs until stopped.

Usage:
    uv run --package historian python -m historian
    uv run --package historian python -m historian --mqtt-host localhost --influx-url http://localhost:8086

The InfluxDB token is read from the INFLUXDB_TOKEN environment variable, so it never
appears in a process list or a container's command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging
from cogniboiler_runtime import run_service

from historian.service import TOKEN_ENV, main
from historian.writer import DEFAULT_TIMEOUT_MS

DEFAULT_METRICS_PORT = 9103


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Historian")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument(
        "--client-id",
        default="historian",
        help="persistent MQTT session id; empty for a clean session",
    )
    parser.add_argument("--influx-url", default="http://localhost:8086")
    parser.add_argument("--influx-org", default="cogniboiler")
    parser.add_argument("--influx-bucket", default="sensors")
    parser.add_argument(
        "--influx-timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_MS / 1000,
        help="how long one InfluxDB write may take",
    )
    parser.add_argument(
        "--aggregate-bucket",
        default="sensors_1m",
        help="one-minute aggregates; empty disables the storage policy",
    )
    parser.add_argument("--raw-retention-days", type=int, default=7)
    parser.add_argument("--aggregate-retention-days", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--flush-interval-s", type=float, default=2.0)
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
    configure_logging("historian")
    args = parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    return run_service(lambda: main(args, token))


if __name__ == "__main__":
    sys.exit(run())
