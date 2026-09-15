"""
Physics Engine entry point.

Runs the live PhysicsService runtime, exposes gRPC state/control APIs,
and optionally mirrors the live state to MQTT.

Usage:
    uv run --package physics-engine python -m physics_engine
    uv run --package physics-engine python -m physics_engine --scenario cold_start
    uv run --package physics-engine python -m physics_engine --scenario steady_state --speed 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add shared/generated to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from physics_engine.async_simulator import ScenarioName
from physics_engine.mqtt_publisher import MQTTConfig, MQTTPublisher
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.server import DEFAULT_PORT as DEFAULT_GRPC_PORT
from physics_engine.server import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("physics_engine")


async def main(
    scenario: ScenarioName,
    speed: float,
    mqtt_host: str,
    mqtt_port: int,
    grpc_port: int,
    enable_mqtt: bool,
) -> None:
    """Run the live simulator runtime, gRPC service, and optional MQTT mirror."""

    runtime = PhysicsRuntime(
        PhysicsRuntimeConfig(
            scenario=scenario,
            speed_factor=speed,
            dt=1.0,
        )
    )

    mqtt_config = MQTTConfig(
        host=mqtt_host,
        port=mqtt_port,
        client_id="physics-engine",
        interval_s=0.1,
    )
    publisher = MQTTPublisher(mqtt_config)

    logger.info(
        "Starting Physics Engine: scenario=%s speed=%.0f× grpc=%d mqtt=%s",
        scenario.value,
        speed,
        grpc_port,
        "enabled" if enable_mqtt else "disabled",
    )

    async def mirror_to_mqtt() -> None:
        last_sequence = -1
        while True:
            try:
                async with publisher.connected() as client:
                    logger.info(
                        "MQTT connected to %s:%d",
                        mqtt_host,
                        mqtt_port,
                    )
                    await publisher.publish_availability(client, "online")
                    while True:
                        last_sequence, state = await runtime.wait_for_update(
                            last_sequence
                        )
                        await publisher.publish_boiler(client, state.boiler)
                        await publisher.publish_turbine(client, state.turbine)
                        await publisher.publish_heartbeat(client)
            except Exception as exc:
                logger.warning("MQTT mirror disconnected: %s", exc)
                await asyncio.sleep(5.0)

    mqtt_task: asyncio.Task[None] | None = None
    if enable_mqtt:
        mqtt_task = asyncio.create_task(mirror_to_mqtt(), name="physics-mqtt-mirror")

    try:
        await serve(runtime, port=grpc_port)
    finally:
        if mqtt_task is not None:
            mqtt_task.cancel()
            try:
                await mqtt_task
            except asyncio.CancelledError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler Physics Engine")
    parser.add_argument(
        "--scenario",
        choices=[s.value for s in ScenarioName],
        default=ScenarioName.STEADY_STATE.value,
        help="Simulation scenario (default: steady_state)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor vs real time (default: 1.0)",
    )
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="PhysicsService gRPC port",
    )
    parser.add_argument(
        "--disable-mqtt",
        action="store_true",
        help="Disable MQTT mirroring and run only the live gRPC service",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            scenario=ScenarioName(args.scenario),
            speed=args.speed,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            grpc_port=args.grpc_port,
            enable_mqtt=not args.disable_mqtt,
        ),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
