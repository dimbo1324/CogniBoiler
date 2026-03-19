"""
Physics Engine entry point.

Runs the AsyncSimulator in the selected scenario and publishes
boiler + turbine state to the MQTT broker via MQTTPublisher.

Usage:
    uv run --package physics-engine python -m physics_engine
    uv run --package physics-engine python -m physics_engine --scenario cold_start
    uv run --package physics-engine python -m physics_engine --scenario steady_state --speed 1
    uv run --package physics-engine python -m physics_engine --scenario steady_state --speed 10 --duration 600
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add shared/generated to path for protobuf imports
sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from physics_engine.async_simulator import AsyncSimulator, ScenarioName, SimulatorConfig
from physics_engine.mqtt_publisher import MQTTConfig, MQTTPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("physics_engine")


async def main(
    scenario: ScenarioName,
    speed: float,
    host: str,
    port: int,
    duration: float,
) -> None:
    """Run simulator and publish telemetry to MQTT."""

    config = SimulatorConfig(
        scenario=scenario,
        speed_factor=speed,
        dt=1.0,
        duration=duration,
    )

    mqtt_config = MQTTConfig(
        host=host,
        port=port,
        client_id="physics-engine",
        interval_s=0.1,
    )

    publisher = MQTTPublisher(mqtt_config)

    logger.info(
        "Starting Physics Engine: scenario=%s speed=%.0f× duration=%.0fs broker=%s:%d",
        scenario.value,
        speed,
        duration,
        host,
        port,
    )

    async with publisher.connected() as client:
        async for state in AsyncSimulator(config).run():
            await publisher.publish_boiler(client, state.boiler)
            await publisher.publish_turbine(client, state.turbine)
            await publisher.publish_heartbeat(client)

            if publisher.published % 100 == 0:
                logger.info(
                    "t=%.0fs  P=%.1f bar  W=%.1f MW  published=%d",
                    state.time,
                    state.boiler.pressure_bar,
                    state.electrical_power_mw,
                    publisher.published,
                )


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
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Scenario duration in seconds (default: 3600)",
    )
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    return parser.parse_args()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    args = parse_args()
    asyncio.run(
        main(
            scenario=ScenarioName(args.scenario),
            speed=args.speed,
            duration=args.duration,
            host=args.host,
            port=args.port,
        )
    )
