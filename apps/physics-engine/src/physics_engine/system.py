"""
Coupled boiler-turbine system model.

Connects BoilerModel and TurbineModel into a single simulation:
    - Boiler produces superheated steam at (T_sh, P_drum)
    - Steam flows through the turbine, generating shaft power
    - Turbine exhaust feeds the condenser (not modelled here)

The coupling point is the steam valve:
    - Boiler sees steam_valve as a flow boundary condition
    - Turbine receives whatever flow the boiler produces at current P, T
"""

from dataclasses import dataclass

from physics_engine.boiler import BoilerModel
from physics_engine.heat_exchanger import SuperheaterModel
from physics_engine.models import BoilerParameters, BoilerState, ControlInputs
from physics_engine.turbine import TurbineModel, TurbineParameters, TurbineState


@dataclass
class SystemState:
    """
    Combined state of the boiler-turbine system at a single time step.

    Aggregates boiler ODE state with instantaneous turbine performance.
    """

    boiler: BoilerState  # full 5D boiler state
    turbine: TurbineState  # instantaneous turbine performance
    time: float  # s — simulation time

    @property
    def electrical_power_mw(self) -> float:
        """Total electrical output [MW]."""
        return self.turbine.electrical_power_mw

    @property
    def steam_flow(self) -> float:
        """Steam mass flow from boiler to turbine [kg/s]."""
        return self.turbine.steam_flow


class BoilerTurbineSystem:
    """
    Coupled boiler-turbine system.

    Runs the boiler ODE simulation and at each requested time point
    evaluates turbine performance using current boiler output conditions.

    Usage:
        system  = BoilerTurbineSystem()
        state   = system.steady_state(
            fuel_valve=0.7,
            feedwater_valve=0.5,
            steam_valve=0.6,
        )
        print(f"Power: {state.electrical_power_mw:.1f} MW")
    """

    def __init__(
        self,
        boiler_params: BoilerParameters | None = None,
        turbine_params: TurbineParameters | None = None,
    ) -> None:
        self.boiler_params = boiler_params or BoilerParameters()
        self.turbine_params = turbine_params or TurbineParameters()

        self.boiler = BoilerModel(self.boiler_params)
        self.turbine = TurbineModel(self.turbine_params)
        self.superheater = SuperheaterModel()

    def _superheated_temp(
        self,
        boiler_state: BoilerState,
        steam_flow: float,
        flue_gas_flow: float,
    ) -> float:
        """
        Calculate superheated steam temperature at turbine inlet [K].

        Uses the SuperheaterModel with current boiler conditions.
        Falls back to drum saturation temperature if steam flow is zero.
        """
        if steam_flow <= 0.0:
            return boiler_state.water_temp

        sh = self.superheater.calculate(
            pressure_pa=boiler_state.pressure,
            steam_flow=steam_flow,
            flue_gas_temp_in=boiler_state.flue_gas_temp,
            flue_gas_flow=flue_gas_flow,
        )
        return sh.steam_temp_out

    def evaluate_at(
        self,
        boiler_state: BoilerState,
        controls: ControlInputs,
        time: float = 0.0,
    ) -> SystemState:
        """
        Evaluate turbine performance given a boiler state snapshot.

        Args:
            boiler_state: Current boiler state from ODE result.
            controls:     Current control inputs (for steam valve position).
            time:         Simulation time [s] (for bookkeeping).

        Returns:
            SystemState combining boiler state and turbine performance.
        """
        # Steam flow through the turbine = boiler steam valve output
        steam_flow = self.boiler._steam_flow(
            boiler_state.pressure,
            controls.steam_valve.position,
        )

        # Flue gas flow from combustion at current fuel valve
        from physics_engine.combustion import CombustionModel

        combustion = CombustionModel(
            max_fuel_flow=self.boiler_params.max_fuel_flow,
        )
        comb = combustion.calculate(fuel_valve=controls.fuel_valve.position)

        # Superheated steam temperature at turbine inlet
        t_steam_in = self._superheated_temp(
            boiler_state, steam_flow, comb.flue_gas_flow
        )

        # Turbine calculation
        turbine_state = self.turbine.calculate(
            steam_temp_in=t_steam_in,
            steam_pressure_in=boiler_state.pressure,
            steam_flow=steam_flow,
        )

        return SystemState(
            boiler=boiler_state,
            turbine=turbine_state,
            time=time,
        )

    def steady_state(
        self,
        fuel_valve: float = 0.7,
        feedwater_valve: float = 0.5,
        steam_valve: float = 0.6,
        t_settle: float = 300.0,
    ) -> SystemState:
        """
        Run boiler to approximate steady state, then evaluate system.

        Simulates the boiler for t_settle seconds with fixed controls
        and returns the system state at the final time point.

        Args:
            fuel_valve:      Fuel valve position [0, 1].
            feedwater_valve: Feedwater valve position [0, 1].
            steam_valve:     Steam valve position [0, 1].
            t_settle:        Settling time [s]. Default 300 s.

        Returns:
            SystemState at end of settling period.
        """
        controls = ControlInputs(
            fuel_valve_command=fuel_valve,
            feedwater_valve_command=feedwater_valve,
            steam_valve_command=steam_valve,
        )
        initial_state = self.boiler_params.nominal_initial_state()
        result = self.boiler.simulate(
            initial_state, controls, t_span=(0.0, t_settle), dt=1.0
        )

        # Take last available time point
        final_index = result.y.shape[1] - 1
        final_boiler_state = self.boiler.get_state_at(result, final_index)
        final_time = float(result.t[final_index])

        return self.evaluate_at(final_boiler_state, controls, time=final_time)
