"""Alert-manager entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from alert_manager.db import init_db
from alert_manager.subscriber import AlertSubscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alert_manager")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Alert Manager")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    return parser.parse_args()


async def main(mqtt_host: str, mqtt_port: int) -> None:
    await init_db()
    subscriber = AlertSubscriber(mqtt_host=mqtt_host, mqtt_port=mqtt_port)
    logger.info("Starting AlertManager: mqtt=%s:%d", mqtt_host, mqtt_port)
    await subscriber.run()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(mqtt_host=args.mqtt_host, mqtt_port=args.mqtt_port),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
