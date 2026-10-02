"""Historian wiring: the MQTT subscriber, the InfluxDB writer, stats and storage policy.

Started by `python -m historian` through `run_service`, so a stop signal cancels `main`
and the batch still buffered is flushed and the InfluxDB client closed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Coroutine
from typing import Any

from cogniboiler_observability import start_metrics_server
from cogniboiler_runtime import LivenessFile

from historian.stats import STATS_INTERVAL_S, report_stats
from historian.storage import StoragePolicy, ensure_storage
from historian.subscriber import HistorianSubscriber
from historian.writer import InfluxWriter

logger = logging.getLogger("historian")

TOKEN_ENV = "INFLUXDB_TOKEN"


async def main(args: argparse.Namespace, influx_token: str) -> None:
    writer = InfluxWriter(
        url=args.influx_url,
        token=influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
        timeout_ms=int(args.influx_timeout_s * 1000),
    )
    subscriber = HistorianSubscriber(
        writer=writer,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        batch_size=args.batch_size,
        flush_interval_s=args.flush_interval_s,
        client_id=args.client_id or None,
        mqtt_username=os.environ.get("MQTT_USERNAME", "historian"),
        mqtt_password=os.environ.get("MQTT_PASSWORD") or None,
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

    tasks: list[Coroutine[Any, Any, None]] = [
        subscriber.run(),
        subscriber.flush_periodically(),
        report_stats(subscriber, writer, STATS_INTERVAL_S),
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
    try:
        await asyncio.gather(*tasks)
    finally:
        try:
            await subscriber.flush()
        finally:
            await asyncio.to_thread(writer.close)
