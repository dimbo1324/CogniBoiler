"""Data models for the CogniBoiler steam boiler simulation."""

from dataclasses import dataclass

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
    TEMP_STEAM_NOMINAL,
    WATER_DENSITY,
)


@dataclass
class BoilerState:
    """
    Represents the instantaneous physical state of the boiler.

    This is the state vector y = [U, P, h, T_gas] that the ODE solver
    integrates over time. All values use SI units.
    """

    internal_energy: float  # J       — total thermal energy stored in water mass
    pressure: float  # Pa      — steam drum pressure
    water_level: float  # m       — water level in drum (0 = empty, drum_height = full)
    flue_gas_temp: float  # K       — flue gas temperature in furnace

    def to_vector(self) -> list[float]:
        """Convert state to a flat list for the ODE solver."""
        return [
            self.internal_energy,
            self.pressure,
            self.water_level,
            self.flue_gas_temp,
        ]

    @classmethod
    def from_vector(cls, y: list[float]) -> "BoilerState":
        """Reconstruct state from ODE solver output vector."""
        return cls(
            internal_energy=y[0],
            pressure=y[1],
            water_level=y[2],
            flue_gas_temp=y[3],
        )

    @property
    def water_temp(self) -> float:
        """
        Estimate water/steam temperature from internal energy.

        Simplified: assumes water mass is constant at nominal,
        uses average specific heat capacity.
        """
        from physics_engine.constants import SPECIFIC_HEAT_WATER, WATER_DENSITY

        water_mass = WATER_DENSITY * DRUM_VOLUME * 0.6  # assume 60% fill
        return self.internal_energy / (water_mass * SPECIFIC_HEAT_WATER)


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

    # Thermal properties
    water_density: float = WATER_DENSITY  # kg/m³
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

        Used to start simulation from a steady-state condition.
        """
        from physics_engine.constants import SPECIFIC_HEAT_WATER

        water_mass = self.water_density * self.drum_volume * 0.6
        u_nominal = water_mass * SPECIFIC_HEAT_WATER * TEMP_STEAM_NOMINAL

        return BoilerState(
            internal_energy=u_nominal,
            pressure=PRESSURE_NOMINAL,
            water_level=self.drum_height * 0.6,
            flue_gas_temp=1273.15,  # K — ~1000°C nominal furnace temperature
        )


@dataclass
class ControlInputs:
    """
    Operator control signals sent to the boiler at each time step.

    Values are normalized: 0.0 = fully closed/off, 1.0 = fully open/max.
    """

    fuel_valve: float = 0.5  # — fuel valve position [0, 1]
    feedwater_valve: float = 0.5  # — feedwater valve position [0, 1]
    steam_valve: float = 0.5  # — steam outlet valve position [0, 1]

    def __post_init__(self) -> None:
        """Clamp all valve positions to [0, 1]."""
        self.fuel_valve = max(0.0, min(1.0, self.fuel_valve))
        self.feedwater_valve = max(0.0, min(1.0, self.feedwater_valve))
        self.steam_valve = max(0.0, min(1.0, self.steam_valve))
