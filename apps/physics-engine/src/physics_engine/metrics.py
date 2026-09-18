"""Prometheus metrics of the physics engine: the step cost, the simulation clock, MQTT."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from physics_engine.runtime import PhysicsRuntime

STEPS = Counter("physics_steps_total", "Plant steps computed.")
STEP_SECONDS = Histogram(
    "physics_step_seconds",
    "Wall-clock time to compute one plant step.",
    buckets=(0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
SIMULATION_TIME = Gauge(
    "physics_simulation_time_seconds", "Simulated time of the current run."
)
SPEED_FACTOR = Gauge(
    "physics_simulation_speed_factor", "Simulated seconds per wall-clock second."
)
RUNNING = Gauge("physics_simulation_running", "1 while the simulation advances.")
ACTIVE_FAULTS = Gauge("physics_active_faults", "Faults active in the plant.")
RUN_ID = Gauge("physics_run_id", "Changes whenever a scenario is loaded.")


def observe_runtime(runtime: PhysicsRuntime) -> None:
    """Read the simulation clock from the runtime whenever Prometheus scrapes."""
    SIMULATION_TIME.set_function(lambda: runtime.snapshot.simulation_time_s)
    SPEED_FACTOR.set_function(lambda: runtime.simulation_status().speed_factor)
    RUNNING.set_function(
        lambda: 1.0 if runtime.simulation_status().run_state.value == "running" else 0.0
    )
    ACTIVE_FAULTS.set_function(lambda: float(len(runtime.snapshot.faults)))
    RUN_ID.set_function(lambda: float(runtime.snapshot.run_id))
