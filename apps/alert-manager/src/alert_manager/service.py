"""Alert-manager wiring: MQTT intake, alarm lifecycle, change publisher and AlarmService.

Started by `python -m alert_manager` through `run_service`, so a stop signal cancels
`main` and the shutdown below runs in dependency order.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Coroutine
from typing import Any

import grpc.aio
from cogniboiler_observability import start_metrics_server
from cogniboiler_runtime import LivenessFile
from sqlalchemy.ext.asyncio import AsyncEngine

from alert_manager.db import create_engine, missing_tables, session_factory
from alert_manager.grpc_server import AlarmServicer, start_server
from alert_manager.processor import AlarmProcessor
from alert_manager.publisher import AlarmChangePublisher
from alert_manager.queries import AlarmQueries
from alert_manager.subscriber import AlertSubscriber

logger = logging.getLogger("alert_manager")


async def main(args: argparse.Namespace) -> int:
    engine = create_engine()
    missing = await missing_tables(engine)
    if missing:
        logger.error(
            "Alarm tables missing: %s — apply the migrations (the migrate job) first",
            ", ".join(missing),
        )
        await engine.dispose()
        return 1

    start_metrics_server(args.metrics_port, args.metrics_host)
    username = os.environ.get("MQTT_USERNAME", "alert-manager")
    password = os.environ.get("MQTT_PASSWORD") or None
    publisher = AlarmChangePublisher(
        args.mqtt_host, args.mqtt_port, username=username, password=password
    )
    sessions = session_factory(engine)
    processor = AlarmProcessor(sessions, publisher)
    subscriber = AlertSubscriber(
        args.mqtt_host, args.mqtt_port, processor, username=username, password=password
    )
    publisher.start()
    server, _ = await start_server(
        AlarmServicer(
            processor,
            AlarmQueries(sessions),
            is_subscribed=lambda: subscriber.connected,
        ),
        args.grpc_port,
    )
    logger.info(
        "Starting AlertManager: mqtt=%s:%d grpc=%d",
        args.mqtt_host,
        args.mqtt_port,
        args.grpc_port,
    )
    tasks: list[Coroutine[Any, Any, None]] = [subscriber.run()]
    if args.liveness_file is not None:
        tasks.append(LivenessFile(args.liveness_file).run(lambda: subscriber.healthy))
    try:
        await asyncio.gather(*tasks)
    finally:
        await _shut_down(server, processor, publisher, engine)
    return 0


async def _shut_down(
    server: grpc.aio.Server,
    processor: AlarmProcessor,
    publisher: AlarmChangePublisher,
    engine: AsyncEngine,
) -> None:
    """Stop in dependency order; a failing step does not skip the ones after it.

    The server goes first so in-flight acknowledgements finish and queue their
    changes, which the publisher then drains before the database is released.
    """
    try:
        await server.stop(grace=5)
        await processor.close()
    finally:
        try:
            await publisher.aclose()
        finally:
            await engine.dispose()
