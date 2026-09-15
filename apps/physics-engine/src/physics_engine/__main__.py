"""
Physics Engine entry point: the live plant, PhysicsService gRPC and the MQTT mirror.

Usage:
    uv run --package physics-engine python -m physics_engine
    uv run --package physics-engine python -m physics_engine --scenario hot_start
    uv run --package physics-engine python -m physics_engine --speed 10 --paused
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

# Add shared/generated to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from physics_engine import properties
from physics_engine.mqtt_publisher import MQTTConfig, MQTTPublisher
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.scenarios import ScenarioName
from physics_engine.server import DEFAULT_PORT, serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("physics_engine")


async def main(args: argparse.Namespace) -> None:
    """Run the live plant, its gRPC service and the optional MQTT mirror."""
    properties.warm_up()
    runtime = PhysicsRuntime(
        PhysicsRuntimeConfig(
            scenario=ScenarioName(args.scenario),
            speed_factor=args.speed,
            dt=args.step_s,
            start_paused=args.paused,
        )
    )
    logger.info(
        "Starting Physics Engine: scenario=%s speed=%g× step=%gs paused=%s grpc=%d mqtt=%s",
        args.scenario,
        args.speed,
        args.step_s,
        args.paused,
        args.grpc_port,
        "disabled" if args.disable_mqtt else f"{args.mqtt_host}:{args.mqtt_port}",
    )

    mirror: asyncio.Task[None] | None = None
    if not args.disable_mqtt:
        publisher = MQTTPublisher(
            MQTTConfig(
                host=args.mqtt_host, port=args.mqtt_port, client_id="physics-engine"
            )
        )
        mirror = asyncio.create_task(
            publisher.mirror_runtime(runtime), name="physics-mqtt-mirror"
        )

    try:
        await serve(runtime, port=args.grpc_port)
    finally:
        if mirror is not None:
            mirror.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mirror


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Physics Engine")
    parser.add_argument(
        "--scenario",
        choices=[s.value for s in ScenarioName],
        default=ScenarioName.STEADY_STATE.value,
        help="scenario to start in (default: steady_state)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="speed vs real time")
    parser.add_argument("--step-s", type=float, default=1.0, help="simulation step [s]")
    parser.add_argument("--paused", action="store_true", help="start paused")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--grpc-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--disable-mqtt", action="store_true", help="gRPC only")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        main(parse_args()),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
