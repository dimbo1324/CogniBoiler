"""
Historian entry point: MQTT telemetry and events into InfluxDB.

Usage:
    uv run --package historian python -m historian
    uv run --package historian python -m historian --mqtt-host localhost --influx-url http://localhost:8086

The InfluxDB token is read from the INFLUXDB_TOKEN environment variable, so it never
appears in a process list or a container's command line.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging, start_metrics_server

from historian.liveness import LivenessFile
from historian.points import build_stats_point
from historian.storage import StoragePolicy, ensure_storage
from historian.subscriber import HistorianSubscriber
from historian.writer import InfluxWriter

logger = logging.getLogger("historian")
DEFAULT_METRICS_PORT = 9103

TOKEN_ENV = "INFLUXDB_TOKEN"
STATS_INTERVAL_S = 30.0


async def main(args: argparse.Namespace, influx_token: str) -> None:
    writer = InfluxWriter(
        url=args.influx_url,
        token=influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
    )
    subscriber = HistorianSubscriber(
        writer=writer,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        batch_size=args.batch_size,
        flush_interval_s=args.flush_interval_s,
        client_id=args.client_id or None,
    )
    logger.info(
        "Starting Historian: mqtt=%s:%d influx=%s bucket=%s aggregates=%s batch=%d",
        args.mqtt_host,
        args.mqtt_port,
        args.influx_url,
        args.influx_bucket,
        args.aggregate_bucket,
        args.batch_size,
    )
    if not influx_token:
        logger.warning("%s is empty: InfluxDB will reject every write", TOKEN_ENV)
    start_metrics_server(args.metrics_port, args.metrics_host)

    async def record_stats() -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL_S)
            stats = subscriber.stats
            logger.info(
                "Stats: received=%d stored=%d skipped=%d writer_errors=%d",
                stats["received"], stats["stored"], stats["skipped"], writer.errors,
            )  # fmt: skip
            point = build_stats_point(stats, writer.errors)
            await asyncio.to_thread(writer.write_point, point)

    tasks: list[Coroutine[Any, Any, None]] = [
        subscriber.run(),
        subscriber.flush_periodically(),
        record_stats(),
    ]
    if args.aggregate_bucket:
        policy = StoragePolicy(
            org=args.influx_org,
            raw_bucket=args.influx_bucket,
            aggregate_bucket=args.aggregate_bucket,
            raw_retention_days=args.raw_retention_days,
            aggregate_retention_days=args.aggregate_retention_days,
        )
        tasks.append(ensure_storage(args.influx_url, influx_token, policy))
    if args.liveness_file is not None:
        tasks.append(LivenessFile(args.liveness_file).run(lambda: subscriber.connected))
    await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging("historian")
    asyncio.run(
        main(parse_args(), os.environ.get(TOKEN_ENV, "")),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
