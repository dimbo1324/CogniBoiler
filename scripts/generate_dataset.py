# scripts/generate_dataset.py
"""
Training dataset generator for CogniBoiler AI/ML pipeline (Phase 6).

Runs all four simulation scenarios, concatenates the results into a
single Pandas DataFrame, and exports it in two formats:

    ml/datasets/raw/cogniboiler_dataset.csv      — human-readable, large
    ml/datasets/raw/cogniboiler_dataset.parquet  — columnar, ~3x smaller

Column schema (all SI units):
    time_s              — simulation time [s]
    scenario            — scenario name (string label)
    pressure_pa         — drum pressure [Pa]
    pressure_bar        — drum pressure [bar]  (convenience)
    water_level_m       — water level [m]
    water_temp_k        — water temperature [K]
    water_temp_c        — water temperature [°C]  (convenience)
    flue_gas_temp_k     — flue gas temperature [K]
    internal_energy_j   — drum internal energy [J]
    electrical_power_w  — generator output [W]
    electrical_power_mw — generator output [MW]  (convenience)
    fuel_valve          — fuel valve position [0–1]
    feedwater_valve     — feedwater valve position [0–1]
    steam_valve         — steam valve position [0–1]
    pressure_error_pa   — PID pressure error [Pa]
    level_error_m       — PID level error [m]

Usage:
    uv run --package physics-engine python scripts/generate_dataset.py

    # Custom output directory:
    uv run --package physics-engine python scripts/generate_dataset.py \
        --output-dir ml/datasets/raw

    # Quick test run (short durations):
    uv run --package physics-engine python scripts/generate_dataset.py \
        --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── sys.path bootstrap ───────────────────────────────────────────────────────
# Add shared/generated to path so cogniboiler_pb2 can be imported.
# This mirrors what conftest.py does for pytest.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "shared" / "generated"))

from physics_engine.scenarios import ScenarioResult, ScenarioRunner  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Scenario definitions ─────────────────────────────────────────────────────

# Each entry: (method_name, kwargs_normal, kwargs_quick)
# quick = short durations for CI / smoke-testing
SCENARIOS: list[tuple[str, dict[str, float], dict[str, float]]] = [
    (
        "steady_state",
        {"duration": 1800.0, "dt": 1.0},  # 30 min nominal
        {"duration": 120.0, "dt": 2.0},  # 2 min quick
    ),
    (
        "load_ramp",
        {"duration": 1800.0, "dt": 1.0},
        {"duration": 120.0, "dt": 2.0},
    ),
    (
        "fuel_trip",
        {"duration": 600.0, "dt": 1.0},
        {"duration": 60.0, "dt": 2.0},
    ),
    (
        "cold_start",
        {"duration": 3600.0, "dt": 2.0},  # 1 hour — longest scenario
        {"duration": 120.0, "dt": 5.0},
    ),
]


# ─── Converter ────────────────────────────────────────────────────────────────


def scenario_result_to_dataframe(result: ScenarioResult) -> pd.DataFrame:
    """
    Convert a ScenarioResult into a Pandas DataFrame.

    Adds convenience columns (bar, °C, MW) alongside raw SI columns
    so downstream ML code does not need to remember unit conversions.

    Args:
        result: ScenarioResult from ScenarioRunner.

    Returns:
        DataFrame with one row per time step.
    """
    n = result.n_steps
    scenario_col = np.full(n, result.scenario_name, dtype=object)

    df = pd.DataFrame(
        {
            "time_s": result.time,
            "scenario": scenario_col,
            # Boiler state — raw SI
            "pressure_pa": result.pressure,
            "pressure_bar": result.pressure / 1.0e5,
            "water_level_m": result.water_level,
            "water_temp_k": result.water_temp,
            "water_temp_c": result.water_temp - 273.15,
            "flue_gas_temp_k": result.flue_gas_temp,
            "internal_energy_j": result.internal_energy,
            # Turbine output
            "electrical_power_w": result.electrical_power,
            "electrical_power_mw": result.electrical_power / 1.0e6,
            # Control signals
            "fuel_valve": result.fuel_valve,
            "feedwater_valve": result.feedwater_valve,
            "steam_valve": result.steam_valve,
            # PID diagnostics
            "pressure_error_pa": result.pressure_error,
            "level_error_m": result.level_error,
        }
    )
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────


def run(output_dir: Path, quick: bool) -> None:
    """
    Run all scenarios and export the combined dataset.

    Args:
        output_dir: Directory where CSV and Parquet files are written.
        quick:      If True, use short durations (fast but sparse data).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = ScenarioRunner()
    frames: list[pd.DataFrame] = []

    total_start = time.perf_counter()

    for method_name, kwargs_normal, kwargs_quick in SCENARIOS:
        kwargs = kwargs_quick if quick else kwargs_normal
        duration = kwargs["duration"]
        dt = kwargs["dt"]
        expected_rows = int(duration / dt)

        logger.info(
            "Running scenario=%-12s  duration=%5.0fs  dt=%.1fs  rows≈%d",
            method_name,
            duration,
            dt,
            expected_rows,
        )

        t_start = time.perf_counter()
        method = getattr(runner, method_name)
        result: ScenarioResult = method(**kwargs)
        elapsed = time.perf_counter() - t_start

        df = scenario_result_to_dataframe(result)
        frames.append(df)

        logger.info(
            "  → done in %.1fs  rows=%d  pressure_range=[%.1f–%.1f] bar",
            elapsed,
            len(df),
            df["pressure_bar"].min(),
            df["pressure_bar"].max(),
        )

    # ── Concatenate all scenarios ─────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)

    # ── Summary statistics ────────────────────────────────────────────────────
    logger.info(
        "Dataset summary: total_rows=%d  scenarios=%d  columns=%d",
        len(combined),
        combined["scenario"].nunique(),
        len(combined.columns),
    )
    logger.info(
        "Rows per scenario:\n%s", combined["scenario"].value_counts().to_string()
    )

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / "cogniboiler_dataset.csv"
    combined.to_csv(csv_path, index=False)
    csv_size_mb = csv_path.stat().st_size / 1_048_576
    logger.info("CSV  written: %s  (%.2f MB)", csv_path, csv_size_mb)

    # ── Export Parquet ────────────────────────────────────────────────────────
    parquet_path = output_dir / "cogniboiler_dataset.parquet"
    combined.to_parquet(parquet_path, index=False, compression="snappy")
    parquet_size_mb = parquet_path.stat().st_size / 1_048_576
    logger.info(
        "Parquet written: %s  (%.2f MB)  compression=%.1f×",
        parquet_path,
        parquet_size_mb,
        csv_size_mb / parquet_size_mb if parquet_size_mb > 0 else 0,
    )

    total_elapsed = time.perf_counter() - total_start
    logger.info("All done in %.1fs", total_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CogniBoiler simulation dataset (CSV + Parquet)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/datasets/raw"),
        help="Output directory for dataset files (default: ml/datasets/raw)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run short scenarios for testing (seconds instead of hours)",
    )
    args = parser.parse_args()
    run(output_dir=args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
