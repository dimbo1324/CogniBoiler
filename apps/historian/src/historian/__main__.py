"""
Historian entry point.

Subscribes to MQTT sensors/# and writes protobuf telemetry
to InfluxDB.

Usage:
    uv run --package historian python -m historian
    uv run --package historian python -m historian --mqtt-host localhost --influx-url http://localhost:8086
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from historian.subscriber import HistorianSubscriber
from historian.writer import InfluxWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("historian")


async def main(
    mqtt_host: str,
    mqtt_port: int,
    influx_url: str,
    influx_token: str,
    influx_org: str,
    influx_bucket: str,
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
    )

    logger.info(
        "Starting Historian: mqtt=%s:%d  influx=%s  bucket=%s",
        mqtt_host,
        mqtt_port,
        influx_url,
        influx_bucket,
    )

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

    await asyncio.gather(
        subscriber.run(),
        log_stats(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Historian")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--influx-url", default="http://localhost:8086")
    parser.add_argument("--influx-token", default="cogniboiler-dev-token")
    parser.add_argument("--influx-org", default="cogniboiler")
    parser.add_argument("--influx-bucket", default="sensors")
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
            influx_url=args.influx_url,
            influx_token=args.influx_token,
            influx_org=args.influx_org,
            influx_bucket=args.influx_bucket,
        )
    )
