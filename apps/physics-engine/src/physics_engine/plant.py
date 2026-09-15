"""
Deterministic plant simulator — the physics engine's single source of process state.

One `step` advances boiler, turbine, condenser, emissions, equipment wear, faults and
instruments by a fixed interval of simulation time. Nothing here reads the wall clock:
the same scenario, commands and faults always produce the same trajectory. The async
runtime paces the simulator against real time; tools and tests drive it directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from physics_engine.boiler import BoilerBalance
from physics_engine.condenser import (
    COOLING_WATER_TEMP_DESIGN,
    CondenserModel,
    CondenserState,
)
from physics_engine.emissions import EmissionsCalculator, EmissionsState
from physics_engine.equipment_health import EquipmentHealth, HealthTracker
from physics_engine.faults import (
    NO_DISTURBANCES,
    ActiveFault,
    FaultError,
    FaultRegistry,
    FaultSpec,
    PlantDisturbances,
    ValveId,
)
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.scenarios import (
    ScenarioDefinition,
    ScenarioName,
    ScheduledFault,
    get_scenario,
)
from physics_engine.sensors import (
    Quality,
    SensorBank,
    SensorId,
    SensorReading,
    worst_quality,
)
from physics_engine.system import BoilerTurbineSystem
from physics_engine.turbine import TurbineParameters, TurbineState

logger = logging.getLogger(__name__)

# Shell and hotwell inertia: back-pressure follows the heat load with this lag.
CONDENSER_TIME_CONSTANT_S: float = 60.0

# Thermal NOx forms in the flame zone, hotter than the gas leaving the furnace.
NOX_FLAME_ZONE_OFFSET_K: float = 300.0

# The turbine counts as on line above this output; a transition starts a run and,
# judged by the steam temperature, costs a cold, warm or hot start of fatigue life.
TURBINE_ONLINE_POWER_W: float = 5.0e6
HOT_START_STEAM_TEMP_K: float = 723.15
WARM_START_STEAM_TEMP_K: float = 473.15


@dataclass(frozen=True)
class PlantConfig:
    """Fixed configuration of a plant simulator."""

    step_s: float = 1.0
    cooling_water_temp_k: float = COOLING_WATER_TEMP_DESIGN
    boiler_params: BoilerParameters = field(default_factory=BoilerParameters)
    turbine_params: TurbineParameters = field(default_factory=TurbineParameters)


@dataclass(frozen=True)
class PlantFlows:
    """Mass flows of the unit [kg/s]."""

    fuel: float
    feedwater: float
    drum_steam: float
    turbine_steam: float
    spray: float
    leak: float
    relief: float


@dataclass(frozen=True)
class PlantHeat:
    """Heat duties [W], temperatures [K] and efficiency of the boiler."""

    heat_release: float
    furnace_to_water: float
    superheater: float
    evaporator_bank: float
    economizer: float
    stack_loss: float
    stack_temp: float
    superheater_outlet_temp: float
    economizer_outlet_temp: float
    feedwater_temp: float
    boiler_efficiency: float


@dataclass(frozen=True)
class PlantSnapshot:
    """Everything known about the plant after a step."""

    simulation_time_s: float
    step_count: int
    step_s: float
    scenario: ScenarioName
    run_id: int
    boiler: BoilerState
    turbine: TurbineState
    controls: ControlInputs
    flows: PlantFlows
    heat: PlantHeat
    emissions: EmissionsState
    condenser: CondenserState
    cooling_water_temp_k: float
    health: EquipmentHealth
    sensors: tuple[SensorReading, ...]
    faults: tuple[ActiveFault, ...]

    def reading(self, sensor: SensorId) -> SensorReading:
        """The reading of one instrument."""
        for reading in self.sensors:
            if reading.sensor_id is sensor:
                return reading
        raise KeyError(sensor)

    def measured(self, sensor: SensorId) -> float:
        """The value an instrument reports."""
        return self.reading(sensor).measured_value

    @property
    def worst_quality(self) -> Quality:
        """Worst quality over all instruments."""
        return worst_quality(self.sensors)

    def measured_boiler(self) -> BoilerState:
        """The boiler state as the instruments report it."""
        return BoilerState(
            internal_energy=self.boiler.internal_energy,
            pressure=self.measured(SensorId.DRUM_PRESSURE),
            water_level=self.measured(SensorId.DRUM_LEVEL),
            flue_gas_temp=self.measured(SensorId.FURNACE_GAS_TEMP),
            water_temp=self.measured(SensorId.DRUM_WATER_TEMP),
        )

    def measured_turbine(self) -> TurbineState:
        """The turbine state with the instrumented values as reported."""
        return replace(
            self.turbine,
            steam_temp_in=self.measured(SensorId.STEAM_TEMP),
            steam_flow=self.measured(SensorId.STEAM_FLOW),
            electrical_power=self.measured(SensorId.ELECTRICAL_POWER),
        )


class PlantSimulator:
    """
    The unit as a deterministic, steppable simulation.

    Usage:
        plant = PlantSimulator(scenario=ScenarioName.STEADY_STATE)
        plant.inject_fault(FaultSpec(FaultKind.STEAM_LEAK, severity=0.05))
        snapshot = plant.step(600)
    """

    def __init__(
        self,
        config: PlantConfig | None = None,
        scenario: ScenarioName | str = ScenarioName.STEADY_STATE,
        *,
        initial_state: BoilerState | None = None,
        initial_controls: ControlInputs | None = None,
    ) -> None:
        self._config = config or PlantConfig()
        if self._config.step_s <= 0.0:
            raise ValueError("step_s must be > 0")
        self._system = BoilerTurbineSystem(
            self._config.boiler_params, self._config.turbine_params
        )
        self._condenser = CondenserModel(
            cooling_water_temp=self._config.cooling_water_temp_k
        )
        self._emissions = EmissionsCalculator()
        self._health = HealthTracker()
        self._sensors = SensorBank()
        self._faults = FaultRegistry()
        self._run_id = 0
        self.load_scenario(
            scenario, initial_state=initial_state, initial_controls=initial_controls
        )

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def config(self) -> PlantConfig:
        return self._config

    @property
    def snapshot(self) -> PlantSnapshot:
        """The plant after the most recent step, scenario load or fault change."""
        return self._snapshot

    @property
    def scenario(self) -> ScenarioDefinition:
        return self._scenario

    @property
    def simulation_time_s(self) -> float:
        return self._step_count * self._config.step_s

    # ─── Scenario ────────────────────────────────────────────────────────────

    def load_scenario(
        self,
        scenario: ScenarioName | str,
        *,
        initial_state: BoilerState | None = None,
        initial_controls: ControlInputs | None = None,
    ) -> PlantSnapshot:
        """Reset the plant into a scenario; faults, wear and time start over."""
        definition = get_scenario(str(scenario))
        conditions = definition.initial_conditions(
            self._config.boiler_params,
            self._config.turbine_params,
            self._config.cooling_water_temp_k,
        )
        self._scenario = definition
        self._state = replace(initial_state or conditions.boiler_state)
        self._controls = (initial_controls or conditions.controls).copy()
        self._exhaust_pressure = conditions.exhaust_pressure_pa
        self._pending_faults: list[ScheduledFault] = sorted(
            definition.scheduled_faults, key=lambda fault: fault.at_s
        )
        self._faults.clear_all()
        self._sensors.reset()
        self._health.reset()
        self._step_count = 0
        self._run_id += 1
        self._turbine_online = False
        self._snapshot = self._evaluate(0.0, NO_DISTURBANCES)
        self._turbine_online = self._snapshot.turbine.electrical_power > (
            TURBINE_ONLINE_POWER_W
        )
        logger.info(
            "Plant loaded scenario=%s run_id=%d", definition.name.value, self._run_id
        )
        return self._snapshot

    # ─── Commands and faults ─────────────────────────────────────────────────

    def apply_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
        spray_valve: float | None = None,
    ) -> None:
        """Set valve commands; the valves travel toward them on the next steps."""
        values = {
            "fuel_valve": fuel_valve,
            "feedwater_valve": feedwater_valve,
            "steam_valve": steam_valve,
        }
        if spray_valve is not None:
            values["spray_valve"] = spray_valve
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value:.3f} outside [0.0, 1.0]")

        self._controls.fuel_valve_command = fuel_valve
        self._controls.feedwater_valve_command = feedwater_valve
        self._controls.steam_valve_command = steam_valve
        if spray_valve is not None:
            self._controls.spray_valve_command = spray_valve

    def inject_fault(self, spec: FaultSpec) -> ActiveFault:
        """Activate a fault now; it shows in the snapshot immediately."""
        fault = self._faults.inject(spec, self.simulation_time_s)
        logger.warning("Fault injected: %s (%s)", fault.label, fault.fault_id)
        self._refresh()
        return fault

    def clear_fault(self, fault_id: str) -> ActiveFault:
        """Deactivate one fault."""
        fault = self._faults.clear(fault_id)
        logger.info("Fault cleared: %s (%s)", fault.label, fault.fault_id)
        self._refresh()
        return fault

    def clear_faults(self) -> tuple[ActiveFault, ...]:
        """Deactivate every fault."""
        cleared = self._faults.clear_all()
        self._refresh()
        return cleared

    # ─── Stepping ────────────────────────────────────────────────────────────

    def step(self, steps: int = 1) -> PlantSnapshot:
        """Advance the plant by `steps` fixed intervals of simulation time."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        dt = self._config.step_s
        for _ in range(steps):
            self._inject_due_faults()
            disturbances = self._faults.disturbances(self.simulation_time_s)
            self._apply_stuck_valves()
            self._state = self._system.boiler.step(
                self._state, self._controls, dt, disturbances
            )
            self._step_count += 1
            self._snapshot = self._evaluate(dt, disturbances)
        return self._snapshot

    def _inject_due_faults(self) -> None:
        now = self.simulation_time_s
        while self._pending_faults and self._pending_faults[0].at_s <= now:
            scheduled = self._pending_faults.pop(0)
            try:
                fault = self._faults.inject(scheduled.spec, now)
            except FaultError as exc:
                logger.warning("Scheduled fault skipped: %s", exc)
                continue
            logger.warning(
                "Scheduled fault injected: %s (%s)", fault.label, fault.fault_id
            )

    def _apply_stuck_valves(self) -> None:
        stuck = self._faults.stuck_valves(self.simulation_time_s)
        valves = dict(
            zip(
                (ValveId.FUEL, ValveId.FEEDWATER, ValveId.STEAM, ValveId.SPRAY),
                self._controls.valves(),
                strict=True,
            )
        )
        for valve_id, valve in valves.items():
            valve.stuck = valve_id in stuck

    def _refresh(self) -> None:
        self._apply_stuck_valves()
        self._snapshot = self._evaluate(
            0.0, self._faults.disturbances(self.simulation_time_s)
        )

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def _evaluate(self, dt: float, disturbances: PlantDisturbances) -> PlantSnapshot:
        state = self._state
        now = self.simulation_time_s
        balance = self._system.boiler.balance(state, self._controls, disturbances)
        turbine = self._system.turbine_at(state, balance, self._exhaust_pressure)

        condenser_target = self._condenser.calculate(
            steam_flow=turbine.steam_flow,
            steam_enthalpy_in=turbine.enthalpy_out_actual,
        )
        if dt > 0.0:
            relax = min(dt / CONDENSER_TIME_CONSTANT_S, 1.0)
            self._exhaust_pressure += (
                condenser_target.backpressure_pa - self._exhaust_pressure
            ) * relax
        condenser = replace(condenser_target, backpressure_pa=self._exhaust_pressure)

        emissions = self._emissions.calculate(
            fuel_flow=balance.fuel_flow,
            flame_temp=state.flue_gas_temp + NOX_FLAME_ZONE_OFFSET_K,
            excess_air_ratio=balance.excess_air_ratio,
        )
        health = self._update_health(dt, state, balance, turbine)

        true_values = {
            SensorId.DRUM_PRESSURE: state.pressure,
            SensorId.DRUM_LEVEL: state.water_level,
            SensorId.DRUM_WATER_TEMP: state.water_temp,
            SensorId.FURNACE_GAS_TEMP: state.flue_gas_temp,
            SensorId.STEAM_TEMP: turbine.steam_temp_in,
            SensorId.STEAM_FLOW: balance.turbine_steam_flow,
            SensorId.FEEDWATER_FLOW: balance.feedwater_flow,
            SensorId.FUEL_FLOW: balance.fuel_flow,
            SensorId.ELECTRICAL_POWER: turbine.electrical_power,
        }
        readings = self._sensors.read(true_values, self._faults.sensor_faults(), now)

        return PlantSnapshot(
            simulation_time_s=now,
            step_count=self._step_count,
            step_s=self._config.step_s,
            scenario=self._scenario.name,
            run_id=self._run_id,
            boiler=replace(state),
            turbine=turbine,
            controls=self._controls.copy(),
            flows=PlantFlows(
                fuel=balance.fuel_flow,
                feedwater=balance.feedwater_flow,
                drum_steam=balance.drum_steam_flow,
                turbine_steam=balance.turbine_steam_flow,
                spray=balance.spray_flow,
                leak=balance.leak_flow,
                relief=balance.relief_flow,
            ),
            heat=PlantHeat(
                heat_release=balance.heat_release,
                furnace_to_water=balance.furnace_to_water,
                superheater=balance.superheater_heat,
                evaporator_bank=balance.evaporator_bank_heat,
                economizer=balance.economizer_heat,
                stack_loss=balance.stack_loss,
                stack_temp=balance.stack_temp,
                superheater_outlet_temp=balance.superheater_outlet_temp,
                economizer_outlet_temp=balance.economizer_outlet_temp,
                feedwater_temp=self._config.boiler_params.feedwater_temp,
                boiler_efficiency=balance.boiler_efficiency,
            ),
            emissions=emissions,
            condenser=condenser,
            cooling_water_temp_k=self._config.cooling_water_temp_k,
            health=health,
            sensors=tuple(readings.values()),
            faults=self._faults.active(),
        )

    def _update_health(
        self,
        dt: float,
        state: BoilerState,
        balance: BoilerBalance,
        turbine: TurbineState,
    ) -> EquipmentHealth:
        if dt <= 0.0:
            return self._health.current_health
        online = turbine.electrical_power > TURBINE_ONLINE_POWER_W
        startup_type = "none"
        if online and not self._turbine_online:
            if turbine.steam_temp_in >= HOT_START_STEAM_TEMP_K:
                startup_type = "hot"
            elif turbine.steam_temp_in >= WARM_START_STEAM_TEMP_K:
                startup_type = "warm"
            else:
                startup_type = "cold"
        self._turbine_online = online
        tube_temp = (
            balance.superheater_outlet_temp
            if balance.drum_steam_flow > 0.0
            else state.water_temp
        )
        return self._health.update(
            dt=dt,
            power_mw=turbine.electrical_power_mw,
            tube_temp_k=tube_temp,
            is_running=online,
            startup_type=startup_type,
        )


__all__ = [
    "PlantConfig",
    "PlantFlows",
    "PlantHeat",
    "PlantSimulator",
    "PlantSnapshot",
    "FaultSpec",
]
