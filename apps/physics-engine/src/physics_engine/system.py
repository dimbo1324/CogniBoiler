"""
Coupled boiler-turbine system model.

Connects BoilerModel and TurbineModel into a single simulation:
    - Boiler produces superheated steam, cooled by attemperator spray
    - Steam flows through the turbine admission valve, generating shaft power
    - Turbine exhaust goes to the condenser (the live plant couples its pressure)

The coupling point is the turbine admission (steam) valve:
    - Boiler sees it as the steam flow boundary condition
    - Turbine receives that flow at the mixed superheater-plus-spray enthalpy
"""

from dataclasses import dataclass

from physics_engine.boiler import BoilerBalance, BoilerModel
from physics_engine.faults import NO_DISTURBANCES, PlantDisturbances
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

    def turbine_at(
        self,
        boiler_state: BoilerState,
        balance: BoilerBalance,
        exhaust_pressure: float | None = None,
    ) -> TurbineState:
        """Turbine performance for a boiler state and its evaluated balance."""
        return self.turbine.calculate_from_enthalpy(
            enthalpy_in=balance.turbine_inlet_enthalpy,
            steam_pressure_in=boiler_state.pressure,
            steam_flow=balance.turbine_steam_flow,
            exhaust_pressure=exhaust_pressure,
        )

    def evaluate_at(
        self,
        boiler_state: BoilerState,
        controls: ControlInputs,
        time: float = 0.0,
        disturbances: PlantDisturbances = NO_DISTURBANCES,
        exhaust_pressure: float | None = None,
    ) -> SystemState:
        """
        Evaluate turbine performance given a boiler state snapshot.

        Args:
            boiler_state:     Current boiler state.
            controls:         Current control inputs (valve positions).
            time:             Simulation time [s] (for bookkeeping).
            disturbances:     Physical effect of active faults.
            exhaust_pressure: Condenser pressure [Pa]; design value if None.

        Returns:
            SystemState combining boiler state and turbine performance.
        """
        balance = self.boiler.balance(boiler_state, controls, disturbances)
        return SystemState(
            boiler=boiler_state,
            turbine=self.turbine_at(boiler_state, balance, exhaust_pressure),
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
