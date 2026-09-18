"""Alert-manager entry point: MQTT alarm intake, alarm lifecycle and AlarmService."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging, start_metrics_server

from alert_manager.db import create_engine, missing_tables, session_factory
from alert_manager.grpc_server import DEFAULT_PORT, AlarmServicer, start_server
from alert_manager.liveness import LivenessFile
from alert_manager.processor import AlarmProcessor
from alert_manager.publisher import AlarmChangePublisher
from alert_manager.subscriber import AlertSubscriber

logger = logging.getLogger("alert_manager")
DEFAULT_METRICS_PORT = 9104


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
    publisher = AlarmChangePublisher(args.mqtt_host, args.mqtt_port)
    processor = AlarmProcessor(session_factory(engine), publisher)
    subscriber = AlertSubscriber(args.mqtt_host, args.mqtt_port, processor)
    publisher.start()
    server = await start_server(
        AlarmServicer(processor, is_subscribed=lambda: subscriber.connected),
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
        tasks.append(LivenessFile(args.liveness_file).run(lambda: subscriber.connected))
    try:
        await asyncio.gather(*tasks)
    finally:
        await server.stop(grace=5)
        await processor.close()
        await publisher.aclose()
        await engine.dispose()
    return 0


if __name__ == "__main__":
    configure_logging("alert-manager")
    sys.exit(
        asyncio.run(
            main(parse_args()),
            # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
            loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
        )
    )
