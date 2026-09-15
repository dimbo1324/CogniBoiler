"""
Coordinated unit control of the drum unit.

Loops, all in SI units:

    Turbine master   electrical power  → turbine admission valve
                     Fast load response from the heat stored in the drum. A pressure
                     guard stops the valve opening, then closes it, when drum pressure
                     falls well below its setpoint, so the turbine cannot drain the boiler.

    Boiler master    drum pressure     → fuel flow demand
                     Feedforward from the ramped load setpoint (the unit heat rate) plus a
                     PI trim; the fuel flow loop turns the demand into a valve position.

    Drum level       three-element     → feedwater flow demand
                     Feedforward from the steam leaving the drum plus a PI trim on level;
                     the feedwater flow loop turns the demand into a valve position.

    Steam temperature                   → attemperator spray valve (reverse acting)

Setpoints move at bounded rates. Priming starts every working setpoint at its measurement
and every loop output at the valve it drives, so taking over the unit — at start-up, on
return from MANUAL or after an E-Stop reset — never bumps an actuator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plc_controller.measurements import (
    SENSOR_DRUM_LEVEL,
    SENSOR_DRUM_PRESSURE,
    SENSOR_ELECTRICAL_POWER,
    SENSOR_STEAM_TEMP,
    ProcessMeasurements,
    ValveSet,
)
from plc_controller.pid import PIDController, PIDParameters
from plc_controller.ramps import RampedSetpoint

# ─── Unit design data the controller is configured with ───────────────────────

RATED_POWER_W: float = 300.0e6
RATED_STEAM_FLOW_KG_S: float = 245.0
FUEL_VALVE_CAPACITY_KG_S: float = 25.0
FEEDWATER_VALVE_CAPACITY_KG_S: float = 380.0

# Heat-rate feedforward: gas flow per watt of load, from the unit's performance curve
# (6.2e-8 at 60 % load to 6.5e-8 at full load); the pressure trim absorbs the rest.
FUEL_PER_WATT: float = 6.4e-8  # kg/s per W

# Below this steam flow there is too little steam to evaporate spray water.
SPRAY_MIN_STEAM_FLOW_KG_S: float = 0.15 * RATED_STEAM_FLOW_KG_S

# Firing follows steam flow: with little steam to cool them, superheater tubes run at
# furnace gas temperature. Fuel is capped at a minimum firing rate — enough to raise
# pressure at start-up — plus what the steam flow can carry away (0.079 kg of gas per
# kg of steam at any load, with margin).
MIN_FIRING_FUEL_KG_S: float = 2.5
FUEL_PER_STEAM_LIMIT: float = 0.1

# ─── Setpoint ramps ───────────────────────────────────────────────────────────

LOAD_RAMP_W_PER_S: float = 0.5e6  # 30 MW/min, 10 %/min — a gas-fired unit's pace
PRESSURE_RAMP_PA_PER_S: float = 5.0e5 / 60.0  # 5 bar/min
LEVEL_RAMP_M_PER_S: float = 0.01
STEAM_TEMP_RAMP_K_PER_S: float = 0.1  # 6 K/min

# ─── Turbine pressure guard ───────────────────────────────────────────────────

PRESSURE_GUARD_HOLD_PA: float = 8.0e5  # deficit at which the valve stops opening
PRESSURE_GUARD_CLOSE_PA: float = 15.0e5  # deficit at which it starts closing
PRESSURE_GUARD_CLOSE_RATE_PER_S: float = 0.01

# A scan that follows a long gap (a lagging stream, a reconnect) must not integrate the
# whole gap at once: the loops are tuned for scans of about one simulation step.
MAX_CONTROL_DT_S: float = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ControlTuning:
    """PID parameters of every loop."""

    turbine_power: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=0.4, ki=0.05, kd=0.0, output_min=0.0, output_max=1.0
        )
    )
    boiler_pressure: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=4.0e-6, ki=5.0e-8, kd=0.0, output_min=-20.0, output_max=20.0
        )
    )
    fuel_flow: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=0.01, ki=0.05, kd=0.0, output_min=-0.3, output_max=0.3
        )
    )
    drum_level: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=120.0, ki=1.5, kd=0.0, output_min=-200.0, output_max=200.0
        )
    )
    feedwater_flow: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=0.001, ki=0.005, kd=0.0, output_min=-0.3, output_max=0.3
        )
    )
    steam_temp: PIDParameters = field(
        default_factory=lambda: PIDParameters(
            kp=-0.004, ki=-0.004, kd=0.0, output_min=0.0, output_max=1.0
        )
    )


@dataclass(frozen=True)
class ControlTargets:
    """What the operator and engineer asked for."""

    load_w: float
    pressure_pa: float
    water_level_m: float
    steam_temp_k: float


@dataclass(frozen=True)
class LoopStatus:
    """One loop at the end of a scan."""

    name: str
    setpoint: float
    measurement: float
    output: float
    unit: str


@dataclass(frozen=True)
class ControlOutput:
    """Valve demands of one scan with the working setpoints and loop signals."""

    valves: ValveSet
    setpoints: ControlTargets
    loops: tuple[LoopStatus, ...]


class UnitController:
    """Coordinated control of boiler and turbine; pure logic, no I/O."""

    def __init__(self, tuning: ControlTuning | None = None) -> None:
        tuning = tuning or ControlTuning()
        self._turbine = PIDController(tuning.turbine_power)
        self._pressure = PIDController(tuning.boiler_pressure)
        self._fuel = PIDController(tuning.fuel_flow)
        self._level = PIDController(tuning.drum_level)
        self._feedwater = PIDController(tuning.feedwater_flow)
        self._steam_temp = PIDController(tuning.steam_temp)
        self.load = RampedSetpoint(LOAD_RAMP_W_PER_S)
        self.pressure = RampedSetpoint(PRESSURE_RAMP_PA_PER_S)
        self.level = RampedSetpoint(LEVEL_RAMP_M_PER_S)
        self.steam_temp = RampedSetpoint(STEAM_TEMP_RAMP_K_PER_S)
        self._primed = False
        self._last_output: ControlOutput | None = None

    @property
    def primed(self) -> bool:
        return self._primed

    @property
    def last_output(self) -> ControlOutput | None:
        return self._last_output

    def invalidate(self) -> None:
        """Force a bumpless re-prime on the next scan (new run, reset, mode change)."""
        self._primed = False

    def working_setpoints(self) -> ControlTargets:
        return ControlTargets(
            load_w=self.load.value,
            pressure_pa=self.pressure.value,
            water_level_m=self.level.value,
            steam_temp_k=self.steam_temp.value,
        )

    # ─── Tracking ────────────────────────────────────────────────────────────

    def track(
        self,
        m: ProcessMeasurements,
        targets: ControlTargets,
        *,
        keep_level: bool = False,
    ) -> None:
        """
        Follow the plant while something else drives the valves (MANUAL, E-Stop).

        Working setpoints sit on the measurements and each loop is loaded with the
        valve it drives, so AUTO can take over at any moment without a bump. With
        `keep_level` the level loops keep running state for `hold_level`.
        """
        keep_level = keep_level and self._primed
        ramps = [
            (self.load, m.electrical_power_w, targets.load_w),
            (self.pressure, m.pressure_pa, targets.pressure_pa),
            (self.steam_temp, m.steam_temp_k, targets.steam_temp_k),
        ]
        if not keep_level:
            ramps.append((self.level, m.water_level_m, targets.water_level_m))
        for ramp, measured, target in ramps:
            ramp.track(measured)
            ramp.set_target(target)

        fuel_demand = m.fuel_flow_kg_s
        self._turbine.reset(initial_output=m.commands.steam)
        self._pressure.reset(
            initial_output=fuel_demand - self.load.value * FUEL_PER_WATT
        )
        self._fuel.reset(
            initial_output=m.commands.fuel - fuel_demand / FUEL_VALVE_CAPACITY_KG_S
        )
        if not keep_level:
            self._level.reset(
                initial_output=m.feedwater_flow_kg_s - m.drum_steam_flow_kg_s
            )
            self._feedwater.reset(
                initial_output=m.commands.feedwater
                - m.feedwater_flow_kg_s / FEEDWATER_VALVE_CAPACITY_KG_S
            )
        self._steam_temp.reset(initial_output=m.commands.spray)
        self._primed = True
        self._last_output = None

    def hold_level(
        self, m: ProcessMeasurements, level_target_m: float, dt: float
    ) -> float:
        """Feedwater valve that keeps the drum at its level while nothing else runs."""
        dt = _clamp(dt, 1.0e-3, MAX_CONTROL_DT_S)
        level_sp = self._ramp(self.level, level_target_m, dt)
        _, valve = self._drum_level(m, level_sp, dt)
        return valve

    # ─── Scan ────────────────────────────────────────────────────────────────

    def scan(
        self, m: ProcessMeasurements, targets: ControlTargets, dt: float
    ) -> ControlOutput:
        """Compute valve demands for one scan of `dt` seconds of plant time."""
        if not self._primed:
            self.track(m, targets)
        dt = _clamp(dt, 1.0e-3, MAX_CONTROL_DT_S)

        load_sp = self._ramp(self.load, targets.load_w, dt)
        pressure_sp = self._ramp(self.pressure, targets.pressure_pa, dt)
        level_sp = self._ramp(self.level, targets.water_level_m, dt)
        steam_temp_sp = self._ramp(self.steam_temp, targets.steam_temp_k, dt)

        steam_valve = self._turbine_master(m, load_sp, pressure_sp, dt)
        fuel_demand, fuel_valve = self._boiler_master(m, load_sp, pressure_sp, dt)
        feedwater_demand, feedwater_valve = self._drum_level(m, level_sp, dt)
        spray_valve = self._steam_temperature(m, steam_temp_sp, dt)

        output = ControlOutput(
            valves=ValveSet(
                fuel=fuel_valve,
                feedwater=feedwater_valve,
                steam=steam_valve,
                spray=spray_valve,
            ),
            setpoints=ControlTargets(
                load_w=load_sp,
                pressure_pa=pressure_sp,
                water_level_m=level_sp,
                steam_temp_k=steam_temp_sp,
            ),
            loops=(
                LoopStatus("load", load_sp, m.electrical_power_w, steam_valve, "W"),
                LoopStatus("pressure", pressure_sp, m.pressure_pa, fuel_demand, "Pa"),
                LoopStatus(
                    "fuel_flow", fuel_demand, m.fuel_flow_kg_s, fuel_valve, "kg/s"
                ),
                LoopStatus(
                    "drum_level", level_sp, m.water_level_m, feedwater_demand, "m"
                ),
                LoopStatus(
                    "feedwater_flow",
                    feedwater_demand,
                    m.feedwater_flow_kg_s,
                    feedwater_valve,
                    "kg/s",
                ),
                LoopStatus(
                    "steam_temp", steam_temp_sp, m.steam_temp_k, spray_valve, "K"
                ),
            ),
        )
        self._last_output = output
        return output

    @staticmethod
    def _ramp(ramp: RampedSetpoint, target: float, dt: float) -> float:
        ramp.set_target(target)
        return ramp.step(dt)

    def _turbine_master(
        self,
        m: ProcessMeasurements,
        load_sp: float,
        pressure_sp: float,
        dt: float,
    ) -> float:
        if m.is_bad(SENSOR_ELECTRICAL_POWER):
            valve = m.commands.steam
        else:
            valve = self._turbine.step(
                setpoint=load_sp / RATED_POWER_W,
                measurement=m.electrical_power_w / RATED_POWER_W,
                dt=dt,
            )

        deficit = pressure_sp - m.pressure_pa
        if deficit > PRESSURE_GUARD_CLOSE_PA:
            valve = min(valve, m.commands.steam - PRESSURE_GUARD_CLOSE_RATE_PER_S * dt)
        elif deficit > PRESSURE_GUARD_HOLD_PA:
            valve = min(valve, m.commands.steam)

        valve = _clamp(valve, 0.0, 1.0)
        self._turbine.constrain(valve)
        return valve

    def _boiler_master(
        self,
        m: ProcessMeasurements,
        load_sp: float,
        pressure_sp: float,
        dt: float,
    ) -> tuple[float, float]:
        feedforward = load_sp * FUEL_PER_WATT
        if m.is_bad(SENSOR_DRUM_PRESSURE):
            trim = self._pressure.state.prev_output
        else:
            trim = self._pressure.step(pressure_sp, m.pressure_pa, dt)
        firing_limit = min(
            FUEL_VALVE_CAPACITY_KG_S,
            MIN_FIRING_FUEL_KG_S + FUEL_PER_STEAM_LIMIT * m.steam_flow_kg_s,
        )
        demand = _clamp(feedforward + trim, 0.0, firing_limit)
        self._pressure.constrain(demand - feedforward)

        base = demand / FUEL_VALVE_CAPACITY_KG_S
        valve = _clamp(base + self._fuel.step(demand, m.fuel_flow_kg_s, dt), 0.0, 1.0)
        self._fuel.constrain(valve - base)
        return demand, valve

    def _drum_level(
        self, m: ProcessMeasurements, level_sp: float, dt: float
    ) -> tuple[float, float]:
        steam = m.drum_steam_flow_kg_s
        if m.is_bad(SENSOR_DRUM_LEVEL):
            trim = self._level.state.prev_output
        else:
            trim = self._level.step(level_sp, m.water_level_m, dt)
        demand = _clamp(steam + trim, 0.0, FEEDWATER_VALVE_CAPACITY_KG_S)
        self._level.constrain(demand - steam)

        base = demand / FEEDWATER_VALVE_CAPACITY_KG_S
        valve = _clamp(
            base + self._feedwater.step(demand, m.feedwater_flow_kg_s, dt), 0.0, 1.0
        )
        self._feedwater.constrain(valve - base)
        return demand, valve

    def _steam_temperature(
        self, m: ProcessMeasurements, steam_temp_sp: float, dt: float
    ) -> float:
        if m.steam_flow_kg_s < SPRAY_MIN_STEAM_FLOW_KG_S:
            self._steam_temp.reset(initial_output=0.0)
            return 0.0
        if m.is_bad(SENSOR_STEAM_TEMP):
            valve = m.commands.spray
            self._steam_temp.reset(initial_output=valve)
            return valve
        return self._steam_temp.step(steam_temp_sp, m.steam_temp_k, dt)
