"""
Historian entry point.

Subscribes to MQTT sensors/# and writes protobuf telemetry
to InfluxDB.

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

from historian.liveness import LivenessFile
from historian.subscriber import HistorianSubscriber
from historian.writer import InfluxWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("historian")

TOKEN_ENV = "INFLUXDB_TOKEN"


async def main(
    mqtt_host: str,
    mqtt_port: int,
    influx_url: str,
    influx_token: str,
    influx_org: str,
    influx_bucket: str,
    batch_size: int,
    flush_interval_s: float,
    liveness_file: Path | None,
) -> None:
    """Connect to MQTT and stream data to InfluxDB."""

    writer = InfluxWriter(
        url=influx_url,
        token=influx_token,
        org=influx_org,
        bucket=influx_bucket,
    )

    subscriber = HistorianSubscriber(
        writer=writer,
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
        batch_size=batch_size,
        flush_interval_s=flush_interval_s,
    )

    logger.info(
        "Starting Historian: mqtt=%s:%d  influx=%s  bucket=%s  batch=%d  flush=%.1fs",
        mqtt_host,
        mqtt_port,
        influx_url,
        influx_bucket,
        batch_size,
        flush_interval_s,
    )
    if not influx_token:
        logger.warning("%s is empty: InfluxDB will reject every write", TOKEN_ENV)

    # Log stats every 30 seconds
    async def log_stats() -> None:
        while True:
            await asyncio.sleep(30)
            stats = subscriber.stats
            logger.info(
                "Stats: received=%d stored=%d skipped=%d  writer_errors=%d",
                stats["received"],
                stats["stored"],
                stats["skipped"],
                writer.errors,
            )

    tasks: list[Coroutine[Any, Any, None]] = [subscriber.run(), log_stats()]
    if liveness_file is not None:
        tasks.append(LivenessFile(liveness_file).run(lambda: subscriber.connected))
    await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Historian")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--influx-url", default="http://localhost:8086")
    parser.add_argument("--influx-org", default="cogniboiler")
    parser.add_argument("--influx-bucket", default="sensors")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--flush-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--liveness-file",
        type=Path,
        default=None,
        help="refresh this file while connected to the broker (container healthcheck)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            influx_url=args.influx_url,
            influx_token=os.environ.get(TOKEN_ENV, ""),
            influx_org=args.influx_org,
            influx_bucket=args.influx_bucket,
            batch_size=args.batch_size,
            flush_interval_s=args.flush_interval_s,
            liveness_file=args.liveness_file,
        ),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
