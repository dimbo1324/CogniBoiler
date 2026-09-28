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
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Add shared/generated to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from cogniboiler_observability import configure_logging, start_metrics_server
from cogniboiler_runtime import run_service

from physics_engine import properties
from physics_engine.metrics import observe_runtime
from physics_engine.mqtt_publisher import MQTTConfig, MQTTPublisher
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.scenarios import ScenarioName
from physics_engine.server import DEFAULT_HOST, DEFAULT_PORT, serve

logger = logging.getLogger("physics_engine")
DEFAULT_METRICS_PORT = 9101


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
    observe_runtime(runtime)
    start_metrics_server(args.metrics_port, args.metrics_host)
    mqtt = "disabled" if args.disable_mqtt else f"{args.mqtt_host}:{args.mqtt_port}"
    logger.info(
        "Starting Physics Engine: scenario=%s speed=%g× step=%gs paused=%s "
        "grpc=%s:%d mqtt=%s",
        *(args.scenario, args.speed, args.step_s, args.paused),
        *(args.grpc_host, args.grpc_port, mqtt),
    )

    mirror: asyncio.Task[None] | None = None
    if not args.disable_mqtt:
        publisher = MQTTPublisher(
            MQTTConfig(
                host=args.mqtt_host,
                port=args.mqtt_port,
                client_id="physics-engine",
                username=os.environ.get("MQTT_USERNAME", "physics-engine"),
                password=os.environ.get("MQTT_PASSWORD") or None,
            )
        )
        mirror = asyncio.create_task(
            publisher.mirror_runtime(runtime), name="physics-mqtt-mirror"
        )

    try:
        await serve(runtime, host=args.grpc_host, port=args.grpc_port)
    finally:
        if mirror is not None:
            mirror.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mirror


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Physics Engine")
    add = parser.add_argument
    add("--scenario", choices=[s.value for s in ScenarioName], default="steady_state")
    add("--speed", type=float, default=1.0, help="speed vs real time")
    add("--step-s", type=float, default=1.0, help="simulation step [s]")
    add("--paused", action="store_true", help="start paused")
    add("--mqtt-host", default="localhost", help="MQTT broker host")
    add("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    add("--grpc-host", default=DEFAULT_HOST, help="0.0.0.0 only in a private network")
    add("--grpc-port", type=int, default=DEFAULT_PORT)
    add("--disable-mqtt", action="store_true", help="gRPC only")
    add("--metrics-port", type=int, default=DEFAULT_METRICS_PORT, help="0 disables")
    add("--metrics-host", default="127.0.0.1", help="interface for /metrics")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    configure_logging("physics-engine")
    args = parse_args(argv)
    sys.exit(run_service(lambda: main(args)))


if __name__ == "__main__":
    run()
