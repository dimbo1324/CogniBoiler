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

from physics_engine.boiler import BoilerBalance, BoilerModel
from physics_engine.models import BoilerParameters, BoilerState
from physics_engine.turbine import TurbineModel, TurbineParameters, TurbineState


class BoilerTurbineSystem:
    """
    Coupled boiler-turbine system.

    Offline settling at fixed valves is `physics_engine.offline.steady_state`.
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
