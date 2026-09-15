"""
Training dataset generator for the deferred CogniBoiler AI stage.

Runs the plant under closed-loop control by the real PLC logic — the deterministic plant
simulator, scanned step by step by PLCService — through a set of episodes: steady
operation, load changes and each kind of fault. Every row carries what the instruments
reported, the true plant state and the labels of the active faults, so a model can later
learn to detect and name faults from measurements alone.

Output (SI units):
    ml/datasets/raw/cogniboiler_dataset.csv
    ml/datasets/raw/cogniboiler_dataset.parquet

Usage:
    uv run --with pandas --with pyarrow python ml/preprocessing/generate_dataset.py
    uv run --with pandas --with pyarrow python ml/preprocessing/generate_dataset.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared" / "generated"))

import cogniboiler_pb2 as pb2  # noqa: E402
from physics_engine.faults import FaultKind, FaultSpec  # noqa: E402
from physics_engine.plant import PlantSimulator, PlantSnapshot  # noqa: E402
from physics_engine.proto_mapping import system_state_to_proto  # noqa: E402
from physics_engine.runtime import RunState, SimulationStatus  # noqa: E402
from physics_engine.scenarios import ScenarioName  # noqa: E402
from physics_engine.sensors import SensorId  # noqa: E402
from plc_controller.service import PLCService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("physics_engine").setLevel(logging.ERROR)
logging.getLogger("plc_controller").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Episode:
    """One closed-loop run: a scenario, operator load changes and injected faults."""

    name: str
    scenario: ScenarioName
    duration_s: int
    load_changes: tuple[tuple[int, float], ...] = ()
    faults: tuple[tuple[int, FaultSpec], ...] = ()


EPISODES: tuple[Episode, ...] = (
    Episode("steady_state", ScenarioName.STEADY_STATE, 1800),
    Episode("load_increase", ScenarioName.PART_LOAD, 1800, ((120, 300.0e6),)),
    Episode("load_decrease", ScenarioName.FULL_LOAD, 1800, ((120, 180.0e6),)),
    Episode(
        "burner_fouling",
        ScenarioName.STEADY_STATE,
        1800,
        faults=((300, FaultSpec(FaultKind.BURNER_FOULING, severity=0.15, ramp_s=600)),),
    ),
    Episode(
        "steam_leak",
        ScenarioName.STEADY_STATE,
        1200,
        faults=((300, FaultSpec(FaultKind.STEAM_LEAK, severity=0.05, ramp_s=120)),),
    ),
    Episode(
        "feedwater_pump_failure",
        ScenarioName.STEADY_STATE,
        900,
        faults=((300, FaultSpec(FaultKind.FEEDWATER_PUMP_FAILURE, severity=0.6)),),
    ),
    Episode(
        "fuel_valve_stuck",
        ScenarioName.STEADY_STATE,
        1200,
        load_changes=((400, 200.0e6),),
        faults=((300, FaultSpec(FaultKind.VALVE_STUCK, target="fuel")),),
    ),
    Episode(
        "level_sensor_drift",
        ScenarioName.STEADY_STATE,
        1200,
        faults=(
            (
                300,
                FaultSpec(FaultKind.SENSOR_DRIFT, target="drum_level", severity=0.01),
            ),
        ),
    ),
    Episode(
        "steam_temp_sensor_failure",
        ScenarioName.STEADY_STATE,
        900,
        faults=((300, FaultSpec(FaultKind.SENSOR_FAILURE, target="steam_temp")),),
    ),
)


class _PlantClient:
    """Stands in for the PhysicsService client: commands go straight to the plant."""

    def __init__(self, plant: PlantSimulator) -> None:
        self._plant = plant

    async def apply_command(self, command: pb2.ControlCommandMsg) -> pb2.CommandAck:
        self._plant.apply_command(
            fuel_valve=command.fuel_valve,
            feedwater_valve=command.feedwater_valve,
            steam_valve=command.steam_valve,
            spray_valve=(
                command.spray_valve if command.HasField("spray_valve") else None
            ),
        )
        return pb2.CommandAck(accepted=True)

    async def health(self) -> pb2.HealthStatus:
        return pb2.HealthStatus(status="running")

    async def close(self) -> None:
        return None


def _status(snapshot: PlantSnapshot) -> SimulationStatus:
    return SimulationStatus(
        run_state=RunState.RUNNING,
        speed_factor=1.0,
        simulation_time_s=snapshot.simulation_time_s,
        step_count=snapshot.step_count,
        scenario=snapshot.scenario,
        run_id=snapshot.run_id,
        step_s=snapshot.step_s,
    )


def _row(episode: str, snapshot: PlantSnapshot, plc_mode: str) -> dict[str, Any]:
    controls = snapshot.controls
    return {
        "episode": episode,
        "scenario": snapshot.scenario.value,
        "time_s": snapshot.simulation_time_s,
        "fault_labels": "|".join(fault.label for fault in snapshot.faults),
        "plc_mode": plc_mode,
        "worst_quality": int(snapshot.worst_quality),
        # What the instruments reported
        "pressure_pa": snapshot.measured(SensorId.DRUM_PRESSURE),
        "water_level_m": snapshot.measured(SensorId.DRUM_LEVEL),
        "water_temp_k": snapshot.measured(SensorId.DRUM_WATER_TEMP),
        "flue_gas_temp_k": snapshot.measured(SensorId.FURNACE_GAS_TEMP),
        "steam_temp_k": snapshot.measured(SensorId.STEAM_TEMP),
        "steam_flow_kg_s": snapshot.measured(SensorId.STEAM_FLOW),
        "feedwater_flow_kg_s": snapshot.measured(SensorId.FEEDWATER_FLOW),
        "fuel_flow_kg_s": snapshot.measured(SensorId.FUEL_FLOW),
        "electrical_power_w": snapshot.measured(SensorId.ELECTRICAL_POWER),
        # The true state
        "true_pressure_pa": snapshot.boiler.pressure,
        "true_water_level_m": snapshot.boiler.water_level,
        "true_steam_temp_k": snapshot.turbine.steam_temp_in,
        "true_electrical_power_w": snapshot.turbine.electrical_power,
        # Actuator commands
        "fuel_valve": controls.fuel_valve_command,
        "feedwater_valve": controls.feedwater_valve_command,
        "steam_valve": controls.steam_valve_command,
        "spray_valve": controls.spray_valve_command,
        # Performance and emissions
        "boiler_efficiency": snapshot.heat.boiler_efficiency,
        "co2_kg_s": snapshot.emissions.co2_rate,
        "nox_ppmv": snapshot.emissions.nox_ppmv,
    }


async def run_episode(episode: Episode, scale: float) -> pd.DataFrame:
    """Run one episode under PLC control and return its rows."""
    plant = PlantSimulator(scenario=episode.scenario)
    plc = PLCService(
        physics_client=_PlantClient(plant),  # type: ignore[arg-type]
        enable_control_loop=False,
        enable_alert_publishing=False,
    )
    load_changes = {int(at * scale): load for at, load in episode.load_changes}
    faults = {int(at * scale): spec for at, spec in episode.faults}

    snapshot = plant.snapshot
    await plc.process_state(system_state_to_proto(snapshot, _status(snapshot)))
    rows: list[dict[str, Any]] = []
    for second in range(int(episode.duration_s * scale)):
        if second in load_changes:
            plc.set_load_demand(load_changes[second], "dataset-generator")
        if second in faults:
            plant.inject_fault(faults[second])
        snapshot = plant.step(1)
        await plc.process_state(system_state_to_proto(snapshot, _status(snapshot)))
        rows.append(_row(episode.name, snapshot, plc.mode.value))
    return pd.DataFrame(rows)


async def run(output_dir: Path, quick: bool) -> None:
    """Run every episode and export the combined dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = 1.0 / 6.0 if quick else 1.0
    frames: list[pd.DataFrame] = []
    started = time.perf_counter()

    for episode in EPISODES:
        episode_started = time.perf_counter()
        frame = await run_episode(episode, scale)
        frames.append(frame)
        logger.info(
            "episode=%-26s rows=%5d  %.1fs  pressure=[%.1f–%.1f] bar",
            episode.name,
            len(frame),
            time.perf_counter() - episode_started,
            frame["pressure_pa"].min() / 1e5,
            frame["pressure_pa"].max() / 1e5,
        )

    combined = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / "cogniboiler_dataset.csv"
    parquet_path = output_dir / "cogniboiler_dataset.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    logger.info(
        "Dataset: %d rows, %d episodes, %d columns in %.1fs -> %s, %s",
        len(combined),
        combined["episode"].nunique(),
        len(combined.columns),
        time.perf_counter() - started,
        csv_path,
        parquet_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the CogniBoiler dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "ml" / "datasets" / "raw",
        help="Directory for the CSV and Parquet files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Episodes at one sixth of their length, for a smoke run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.output_dir, args.quick))
