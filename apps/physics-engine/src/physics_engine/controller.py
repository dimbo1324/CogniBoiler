"""
Boiler controller: three PID control loops for drum pressure,
drum water level, and superheated steam temperature.

Control architecture:
    Loop 1 — Pressure:
        Master: pressure PID     setpoint: bar -> output: fuel flow setpoint [kg/s]
        Slave:  fuel flow PID    setpoint: kg/s -> output: fuel valve command [0, 1]

    Loop 2 — Drum level:
        Master: level PID        setpoint: m -> output: feedwater flow setpoint [kg/s]
        Slave:  feedwater PID    setpoint: kg/s -> output: feedwater valve command [0, 1]

    Loop 3 — Steam temperature:
        Single PID               setpoint: K -> output: steam valve command [0, 1]
        (simple throttling — spray injection modelled in Phase 3.2)

All loops support AUTO / MANUAL mode with bumpless transfer.
"""

from dataclasses import dataclass

from physics_engine.pid import (
    CascadePIDController,
    CascadePIDParameters,
    PIDController,
    PIDParameters,
)

# ─── Operating mode ───────────────────────────────────────────────────────────


class ControlMode:
    """Controller operating mode constants."""

    AUTO: str = "AUTO"
    MANUAL: str = "MANUAL"


# ─── Setpoints ────────────────────────────────────────────────────────────────


@dataclass
class BoilerSetpoints:
    """
    Operator setpoints for all three control loops.

    These are the desired operating targets.  The controller drives
    the boiler toward these values automatically in AUTO mode.
    """

    pressure: float = 140.0e5  # Pa  — drum pressure setpoint (140 bar)
    water_level: float = 4.8  # m   — drum water level setpoint (60% of 8 m)
    steam_temp: float = 825.65  # K   — superheated steam temp setpoint (552.5°C)


# ─── Controller output ────────────────────────────────────────────────────────


@dataclass
class ControllerOutput:
    """
    Valve commands produced by the controller at each time step.

    All values normalized [0, 1].
    """

    fuel_valve: float  # — fuel valve command
    feedwater_valve: float  # — feedwater valve command
    steam_valve: float  # — steam valve command

    # Intermediate signals (useful for diagnostics and logging)
    fuel_flow_setpoint: float = 0.0  # kg/s — master pressure loop output
    feedwater_flow_setpoint: float = 0.0  # kg/s — master level loop output
    pressure_error: float = 0.0  # Pa  — current pressure error
    level_error: float = 0.0  # m   — current level error
    temp_error: float = 0.0  # K   — current temperature error


# ─── Default PID tunings ──────────────────────────────────────────────────────

# Loop 1 — Pressure cascade
# Master: pressure [Pa] -> fuel flow setpoint [kg/s]
#   Large Kp needed — pressure range is 100–180 bar = 1e7 Pa
#   Output range: 0–10 kg/s (max fuel flow)
_PRESSURE_MASTER = PIDParameters(
    kp=5e-6,  # 1e5 Pa error -> 0.5 kg/s fuel change
    ki=1e-7,  # slow integral — pressure is sluggish
    kd=1e-5,  # mild derivative — damps pressure oscillations
    output_min=0.0,
    output_max=10.0,  # max fuel flow [kg/s]
    tau_d=5.0,  # 5 s derivative filter — rejects sensor noise
    anti_windup=True,
)

# Slave: fuel flow [kg/s] -> fuel valve command [0, 1]
_PRESSURE_SLAVE = PIDParameters(
    kp=0.08,  # 1 kg/s error -> 0.08 valve movement
    ki=0.02,  # moderate integral
    kd=0.01,
    output_min=0.0,
    output_max=1.0,
    tau_d=1.0,
    anti_windup=True,
)

# Loop 2 — Level cascade
# Master: level [m] -> feedwater flow setpoint [kg/s]
#   Level range: 0–8 m, nominal 4.8 m
#   Output range: 0–300 kg/s (max feedwater flow)
_LEVEL_MASTER = PIDParameters(
    kp=30.0,  # 1 m error -> 30 kg/s feedwater change
    ki=2.0,  # moderate integral
    kd=5.0,  # derivative helps with swell/shrink effect
    output_min=0.0,
    output_max=300.0,  # max feedwater flow [kg/s]
    tau_d=3.0,
    anti_windup=True,
)

# Slave: feedwater flow [kg/s] -> feedwater valve command [0, 1]
_LEVEL_SLAVE = PIDParameters(
    kp=0.003,  # 1 kg/s error -> 0.003 valve movement
    ki=0.001,
    kd=0.0,
    output_min=0.0,
    output_max=1.0,
    tau_d=0.0,
    anti_windup=True,
)

# Loop 3 — Steam temperature (single PID)
# Steam temp [K] -> steam valve command [0, 1]
# Logic: if temp too high -> open steam valve more (release hot steam)
#        if temp too low  -> close steam valve (let steam superheat longer)
_STEAM_TEMP = PIDParameters(
    kp=0.002,  # 1 K error -> 0.002 valve movement
    ki=0.0001,
    kd=0.005,
    output_min=0.0,
    output_max=1.0,
    tau_d=10.0,  # heavy filtering — temperature is very slow
    anti_windup=True,
)


# ─── Boiler controller ────────────────────────────────────────────────────────


class BoilerController:
    """
    Three-loop boiler controller.

    Manages pressure, drum level, and steam temperature using
    cascade PID (loops 1–2) and single PID (loop 3).

    All loops independently switchable between AUTO and MANUAL.

    Usage:
        controller = BoilerController()
        setpoints  = BoilerSetpoints(pressure=140e5, water_level=4.8)

        # In simulation loop:
        output = controller.step(
            setpoints=setpoints,
            pressure=state.pressure,
            water_level=state.water_level,
            steam_temp=state.water_temp,
            fuel_flow=comb.fuel_flow,
            feedwater_flow=feedwater_flow,
            dt=1.0,
        )
        controls = ControlInputs(
            fuel_valve_command=output.fuel_valve,
            feedwater_valve_command=output.feedwater_valve,
            steam_valve_command=output.steam_valve,
        )
    """

    def __init__(
        self,
        pressure_params: CascadePIDParameters | None = None,
        level_params: CascadePIDParameters | None = None,
        temp_params: PIDParameters | None = None,
    ) -> None:
        # Loop 1 — pressure
        self.pressure_loop = CascadePIDController(
            pressure_params
            or CascadePIDParameters(
                master=_PRESSURE_MASTER,
                slave=_PRESSURE_SLAVE,
                slave_setpoint_min=0.0,
                slave_setpoint_max=10.0,
            )
        )

        # Loop 2 — drum level
        self.level_loop = CascadePIDController(
            level_params
            or CascadePIDParameters(
                master=_LEVEL_MASTER,
                slave=_LEVEL_SLAVE,
                slave_setpoint_min=0.0,
                slave_setpoint_max=300.0,
            )
        )

        # Loop 3 — steam temperature
        self.temp_loop = PIDController(temp_params or _STEAM_TEMP)

        # Operating modes per loop
        self._pressure_mode: str = ControlMode.AUTO
        self._level_mode: str = ControlMode.AUTO
        self._temp_mode: str = ControlMode.AUTO

    # ─── Mode management ──────────────────────────────────────────────────────

    def set_pressure_manual(self, fuel_valve: float) -> None:
        """Hold fuel valve at fixed position."""
        self.pressure_loop.set_manual(fuel_valve)
        self._pressure_mode = ControlMode.MANUAL

    def set_level_manual(self, feedwater_valve: float) -> None:
        """Hold feedwater valve at fixed position."""
        self.level_loop.set_manual(feedwater_valve)
        self._level_mode = ControlMode.MANUAL

    def set_temp_manual(self, steam_valve: float) -> None:
        """Hold steam valve at fixed position."""
        self.temp_loop.set_manual(steam_valve)
        self._temp_mode = ControlMode.MANUAL

    def set_all_auto(self) -> None:
        """Switch all loops to AUTO."""
        self.pressure_loop.set_auto()
        self.level_loop.set_auto()
        self.temp_loop.set_auto()
        self._pressure_mode = ControlMode.AUTO
        self._level_mode = ControlMode.AUTO
        self._temp_mode = ControlMode.AUTO

    def set_all_manual(
        self,
        fuel_valve: float = 0.5,
        feedwater_valve: float = 0.5,
        steam_valve: float = 0.5,
    ) -> None:
        """Switch all loops to MANUAL with given fixed outputs."""
        self.set_pressure_manual(fuel_valve)
        self.set_level_manual(feedwater_valve)
        self.set_temp_manual(steam_valve)

    # ─── Main step ────────────────────────────────────────────────────────────

    def step(
        self,
        setpoints: BoilerSetpoints,
        pressure: float,
        water_level: float,
        steam_temp: float,
        fuel_flow: float,
        feedwater_flow: float,
        dt: float,
    ) -> ControllerOutput:
        """
        Compute controller outputs for one time step.

        Args:
            setpoints:       Desired operating targets.
            pressure:        Measured drum pressure [Pa].
            water_level:     Measured drum water level [m].
            steam_temp:      Measured superheated steam temperature [K].
            fuel_flow:       Measured fuel mass flow [kg/s] (inner loop feedback).
            feedwater_flow:  Measured feedwater mass flow [kg/s] (inner loop feedback).
            dt:              Time step [s].

        Returns:
            ControllerOutput with valve commands and diagnostic signals.
        """
        # ── Loop 1: pressure -> fuel valve ────────────────────────────────────
        fuel_valve_cmd = self.pressure_loop.step(
            primary_setpoint=setpoints.pressure,
            primary_measurement=pressure,
            inner_measurement=fuel_flow,
            dt=dt,
        )

        # ── Loop 2: level -> feedwater valve ──────────────────────────────────
        feedwater_valve_cmd = self.level_loop.step(
            primary_setpoint=setpoints.water_level,
            primary_measurement=water_level,
            inner_measurement=feedwater_flow,
            dt=dt,
        )

        # ── Loop 3: steam temp -> steam valve ─────────────────────────────────
        steam_valve_cmd = self.temp_loop.step(
            setpoint=setpoints.steam_temp,
            measurement=steam_temp,
            dt=dt,
        )

        return ControllerOutput(
            fuel_valve=fuel_valve_cmd,
            feedwater_valve=feedwater_valve_cmd,
            steam_valve=steam_valve_cmd,
            fuel_flow_setpoint=self.pressure_loop.master.state.prev_output,
            feedwater_flow_setpoint=self.level_loop.master.state.prev_output,
            pressure_error=setpoints.pressure - pressure,
            level_error=setpoints.water_level - water_level,
            temp_error=setpoints.steam_temp - steam_temp,
        )

    def reset(self) -> None:
        """Reset all controllers to initial state."""
        self.pressure_loop.reset()
        self.level_loop.reset()
        self.temp_loop.reset()
