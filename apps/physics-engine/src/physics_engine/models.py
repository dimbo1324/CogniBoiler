"""Data models for the CogniBoiler steam boiler simulation."""

from dataclasses import dataclass, field

from physics_engine import steam_tables
from physics_engine.constants import (
    DRUM_CROSS_SECTION,
    DRUM_HEIGHT,
    DRUM_VOLUME,
    HEAT_LOSS_COEFFICIENT,
    HEAT_TRANSFER_GAS_WATER,
    MAX_FUEL_FLOW,
    MAX_STEAM_FLOW,
    MIN_STEAM_FLOW,
    PRESSURE_NOMINAL,
    STEAM_VALVE_COEFFICIENT,
    TEMP_AMBIENT,
    TEMP_FEEDWATER,
)


@dataclass
class BoilerState:
    """
    Represents the instantaneous physical state of the boiler.

    State vector y = [U, P, h, T_gas, T_water] — 5 dimensions.
    T_water is now explicit to avoid repeated recomputation from U
    and to eliminate numerical drift in the energy-to-temperature inversion.

    All values use SI units.
    """

    internal_energy: float  # J   — total thermal energy stored in water mass
    pressure: float  # Pa  — steam drum pressure
    water_level: float  # m   — water level in drum (0=empty, drum_height=full)
    flue_gas_temp: float  # K   — flue gas temperature in furnace
    water_temp: float  # K   — bulk water/steam temperature in drum

    def to_vector(self) -> list[float]:
        """Convert state to a flat list for the ODE solver."""
        return [
            self.internal_energy,
            self.pressure,
            self.water_level,
            self.flue_gas_temp,
            self.water_temp,
        ]

    @classmethod
    def from_vector(cls, y: list[float]) -> "BoilerState":
        """Reconstruct state from ODE solver output vector."""
        return cls(
            internal_energy=y[0],
            pressure=y[1],
            water_level=y[2],
            flue_gas_temp=y[3],
            water_temp=y[4],
        )

    @property
    def pressure_bar(self) -> float:
        """Pressure in bar (convenience property)."""
        return self.pressure / 1.0e5

    @property
    def water_temp_celsius(self) -> float:
        """Water temperature in Celsius (convenience property)."""
        return self.water_temp - 273.15

    @property
    def flue_gas_temp_celsius(self) -> float:
        """Flue gas temperature in Celsius (convenience property)."""
        return self.flue_gas_temp - 273.15


@dataclass
class ValveState:
    """
    Physical state of a control valve including actuation dynamics.

    Real valves do not jump instantly to a commanded position —
    they move at a finite rate (typical: 10–20% per second for
    motorized valves in power plant service).

    This prevents instantaneous step changes in flow that would
    create numerical stiffness in the ODE solver and are physically
    unrealistic.
    """

    position: float  # —    current actual valve opening [0, 1]
    command: float  # —    operator commanded position [0, 1]
    rate_limit: float = 0.15  # 1/s  maximum rate of change (15% per second)

    def __post_init__(self) -> None:
        self.position = max(0.0, min(1.0, self.position))
        self.command = max(0.0, min(1.0, self.command))

    def step(self, dt: float) -> None:
        """
        Advance valve position toward command by one time step.

        Args:
            dt: Time step [s].
        """
        max_move = self.rate_limit * dt
        error = self.command - self.position
        move = max(-max_move, min(max_move, error))
        self.position = max(0.0, min(1.0, self.position + move))


@dataclass
class ControlInputs:
    """
    Operator control commands sent to the boiler.

    Contains both the commanded setpoints and the actual valve states
    (which lag behind commands due to actuator dynamics).

    Values are normalized: 0.0 = fully closed/off, 1.0 = fully open/max.
    """

    fuel_valve_command: float = 0.5  # — commanded fuel valve position
    feedwater_valve_command: float = 0.5  # — commanded feedwater valve position
    steam_valve_command: float = 0.5  # — commanded steam valve position

    # Actual valve states with dynamics (initialized at command position)
    fuel_valve: ValveState = field(init=False)
    feedwater_valve: ValveState = field(init=False)
    steam_valve: ValveState = field(init=False)

    def __post_init__(self) -> None:
        self.fuel_valve_command = max(0.0, min(1.0, self.fuel_valve_command))
        self.feedwater_valve_command = max(0.0, min(1.0, self.feedwater_valve_command))
        self.steam_valve_command = max(0.0, min(1.0, self.steam_valve_command))

        # Initialise valves at current command positions
        self.fuel_valve = ValveState(
            position=self.fuel_valve_command,
            command=self.fuel_valve_command,
        )
        self.feedwater_valve = ValveState(
            position=self.feedwater_valve_command,
            command=self.feedwater_valve_command,
        )
        self.steam_valve = ValveState(
            position=self.steam_valve_command,
            command=self.steam_valve_command,
        )

    def update_valves(self, dt: float) -> None:
        """Advance all valve positions by one time step."""
        self.fuel_valve.command = self.fuel_valve_command
        self.feedwater_valve.command = self.feedwater_valve_command
        self.steam_valve.command = self.steam_valve_command

        self.fuel_valve.step(dt)
        self.feedwater_valve.step(dt)
        self.steam_valve.step(dt)


@dataclass
class BoilerParameters:
    """
    Fixed configuration parameters for a boiler instance.

    These do not change during simulation — they describe the physical
    design of the specific boiler unit.
    """

    # Geometry
    drum_volume: float = DRUM_VOLUME  # m³
    drum_cross_section: float = DRUM_CROSS_SECTION  # m²
    drum_height: float = DRUM_HEIGHT  # m

    # Thermal properties (fallback — actual values from steam_tables)
    heat_loss_coeff: float = HEAT_LOSS_COEFFICIENT  # W/K
    heat_transfer_coeff: float = HEAT_TRANSFER_GAS_WATER  # W/K

    # Operating limits
    max_fuel_flow: float = MAX_FUEL_FLOW  # kg/s
    max_steam_flow: float = MAX_STEAM_FLOW  # kg/s
    min_steam_flow: float = MIN_STEAM_FLOW  # kg/s
    steam_valve_coeff: float = STEAM_VALVE_COEFFICIENT  # kg/(s·bar)

    # Boundary conditions
    feedwater_temp: float = TEMP_FEEDWATER  # K
    ambient_temp: float = TEMP_AMBIENT  # K

    def nominal_initial_state(self) -> "BoilerState":
        """
        Return a physically consistent initial state at nominal operating point.
        """
        water_level_nominal = self.drum_height * 0.6

        t_sat = steam_tables.saturation_temp(PRESSURE_NOMINAL)

        water_density = steam_tables.water_density(
            temp_k=t_sat,
            pressure_pa=PRESSURE_NOMINAL,
        )
        water_mass = water_density * self.drum_cross_section * water_level_nominal
        u_nominal = water_mass * steam_tables.water_enthalpy(t_sat, PRESSURE_NOMINAL)

        return BoilerState(
            internal_energy=u_nominal,
            pressure=PRESSURE_NOMINAL,
            water_level=water_level_nominal,
            flue_gas_temp=1273.15,  # K — ~1000°C nominal furnace temperature
            water_temp=t_sat,
        )
