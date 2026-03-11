"""
Thermodynamic model of a steam turbine stage.

Models isentropic expansion of superheated steam through a turbine,
calculating shaft power output and exhaust steam conditions.

Physics:
    W_ideal  = h_in − h_out_isentropic          [J/kg]
    W_actual = η_is × W_ideal                   [J/kg]
    P_shaft  = m_steam × W_actual               [W]
    h_out    = h_in − W_actual                  [J/kg]

Isentropic efficiency η_is accounts for:
    - Blade friction and aerodynamic losses
    - Steam leakage through seals
    - Mechanical friction in bearings
    Typical range for HP turbines: 0.85–0.92
"""

from dataclasses import dataclass

from physics_engine import steam_tables
from physics_engine.constants import (
    PRESSURE_NOMINAL,
    TEMP_STEAM_NOMINAL,
)

# ─── Turbine design constants ──────────────────────────────────────────────────

# Isentropic efficiency of the turbine [-]
# 0.88 is representative of a modern HP steam turbine at design point.
TURBINE_ISENTROPIC_EFFICIENCY: float = 0.88

# Turbine exhaust (condenser) pressure [Pa]
# Typical condensing turbine back-pressure: 0.05–0.10 bar absolute.
# 0.07 bar = 7 kPa is a common design value.
TURBINE_EXHAUST_PRESSURE: float = 7_000.0  # Pa  (0.07 bar)

# Mechanical efficiency of turbine-generator coupling [-]
# Accounts for bearing friction and gear losses.
TURBINE_MECHANICAL_EFFICIENCY: float = 0.98

# Minimum steam flow to sustain rotation [kg/s]
# Below this, turbine is considered offline.
TURBINE_MIN_STEAM_FLOW: float = 10.0  # kg/s

# Nominal steam inlet conditions (matches boiler superheater output)
TURBINE_NOMINAL_INLET_PRESSURE: float = PRESSURE_NOMINAL  # 140 bar
TURBINE_NOMINAL_INLET_TEMP: float = TEMP_STEAM_NOMINAL  # 825.65 K / 552.5°C


@dataclass
class TurbineState:
    """
    Operating state of the turbine at a single time step.

    All values in SI units.
    """

    # Inlet conditions
    steam_temp_in: float  # K      — steam temperature at turbine inlet
    steam_pressure_in: float  # Pa     — steam pressure at turbine inlet
    steam_flow: float  # kg/s   — steam mass flow through turbine

    # Thermodynamic quantities
    enthalpy_in: float  # J/kg   — specific enthalpy at inlet
    entropy_in: float  # J/(kg·K) — specific entropy at inlet
    enthalpy_out_isentropic: float  # J/kg — enthalpy at ideal (isentropic) outlet
    enthalpy_out_actual: float  # J/kg   — enthalpy at real outlet

    # Work and power
    specific_work_ideal: float  # J/kg   — isentropic specific work
    specific_work_actual: float  # J/kg   — actual specific work
    shaft_power: float  # W      — mechanical power at shaft
    electrical_power: float  # W      — power after mechanical losses

    # Outlet conditions
    exhaust_pressure: float  # Pa     — turbine exhaust pressure
    exhaust_temp: float  # K      — exhaust steam temperature

    # Efficiency metrics
    isentropic_efficiency: float  # —    — η_is used in this calculation

    @property
    def shaft_power_mw(self) -> float:
        """Shaft power in megawatts (convenience property)."""
        return self.shaft_power / 1.0e6

    @property
    def electrical_power_mw(self) -> float:
        """Electrical power output in megawatts (convenience property)."""
        return self.electrical_power / 1.0e6


@dataclass
class TurbineParameters:
    """
    Fixed design parameters for a turbine instance.

    These describe the physical design and do not change during simulation.
    """

    isentropic_efficiency: float = TURBINE_ISENTROPIC_EFFICIENCY
    mechanical_efficiency: float = TURBINE_MECHANICAL_EFFICIENCY
    exhaust_pressure: float = TURBINE_EXHAUST_PRESSURE  # Pa
    min_steam_flow: float = TURBINE_MIN_STEAM_FLOW  # kg/s
    nominal_inlet_pressure: float = TURBINE_NOMINAL_INLET_PRESSURE  # Pa
    nominal_inlet_temp: float = TURBINE_NOMINAL_INLET_TEMP  # K


class TurbineModel:
    """
    Isentropic expansion model of a steam turbine.

    Given inlet steam conditions (T, P) and mass flow, calculates:
        - Shaft power output
        - Electrical power output (after mechanical losses)
        - Exhaust steam conditions

    Usage:
        params  = TurbineParameters()
        turbine = TurbineModel(params)
        state   = turbine.calculate(
            steam_temp_in=825.65,
            steam_pressure_in=140e5,
            steam_flow=200.0,
        )
        print(f"Power: {state.electrical_power_mw:.1f} MW")
    """

    def __init__(self, params: TurbineParameters | None = None) -> None:
        self.params = params or TurbineParameters()

    def calculate(
        self,
        steam_temp_in: float,
        steam_pressure_in: float,
        steam_flow: float,
    ) -> TurbineState:
        """
        Calculate turbine performance at given inlet conditions.

        Args:
            steam_temp_in:      Steam temperature at turbine inlet [K].
            steam_pressure_in:  Steam pressure at turbine inlet [Pa].
            steam_flow:         Steam mass flow rate [kg/s].

        Returns:
            TurbineState with all calculated thermodynamic quantities.
        """
        p_out = self.params.exhaust_pressure

        # ── Turbine offline — no steam flow ───────────────────────────────────
        if steam_flow < self.params.min_steam_flow:
            h_in = steam_tables.steam_enthalpy(steam_temp_in, steam_pressure_in)
            s_in = steam_tables.steam_entropy(steam_temp_in, steam_pressure_in)
            h_out_is = steam_tables.isentropic_enthalpy(s_in, p_out)
            return TurbineState(
                steam_temp_in=steam_temp_in,
                steam_pressure_in=steam_pressure_in,
                steam_flow=0.0,
                enthalpy_in=h_in,
                entropy_in=s_in,
                enthalpy_out_isentropic=h_out_is,
                enthalpy_out_actual=h_in,  # no expansion — outlet = inlet
                specific_work_ideal=0.0,
                specific_work_actual=0.0,
                shaft_power=0.0,
                electrical_power=0.0,
                exhaust_pressure=p_out,
                exhaust_temp=steam_temp_in,
                isentropic_efficiency=self.params.isentropic_efficiency,
            )

        # ── Inlet thermodynamic state ─────────────────────────────────────────
        h_in = steam_tables.steam_enthalpy(steam_temp_in, steam_pressure_in)
        s_in = steam_tables.steam_entropy(steam_temp_in, steam_pressure_in)

        # ── Isentropic outlet enthalpy ────────────────────────────────────────
        # h at (s=s_in, P=P_out) — the ideal expansion endpoint
        h_out_isentropic = steam_tables.isentropic_enthalpy(s_in, p_out)

        # ── Actual outlet enthalpy ────────────────────────────────────────────
        # η_is < 1 means less work extracted, so h_out_actual > h_out_isentropic
        w_ideal = h_in - h_out_isentropic
        w_ideal = max(w_ideal, 0.0)  # cannot extract negative work

        w_actual = self.params.isentropic_efficiency * w_ideal
        h_out_actual = h_in - w_actual

        # ── Power ─────────────────────────────────────────────────────────────
        p_shaft = steam_flow * w_actual
        p_electrical = p_shaft * self.params.mechanical_efficiency

        # ── Exhaust temperature ───────────────────────────────────────────────
        # Find T at (h=h_out_actual, P=P_out) via IAPWS-IF97
        t_exhaust = steam_tables.exhaust_temp(h_out_actual, p_out)

        return TurbineState(
            steam_temp_in=steam_temp_in,
            steam_pressure_in=steam_pressure_in,
            steam_flow=steam_flow,
            enthalpy_in=h_in,
            entropy_in=s_in,
            enthalpy_out_isentropic=h_out_isentropic,
            enthalpy_out_actual=h_out_actual,
            specific_work_ideal=w_ideal,
            specific_work_actual=w_actual,
            shaft_power=p_shaft,
            electrical_power=p_electrical,
            exhaust_pressure=p_out,
            exhaust_temp=t_exhaust,
            isentropic_efficiency=self.params.isentropic_efficiency,
        )

    def nominal_state(self) -> TurbineState:
        """
        Calculate turbine state at nominal design point.

        Uses nominal boiler output: 552.5°C, 140 bar, 277.8 kg/s.
        Useful for sanity-checking: expected output ~220–260 MW.
        """
        from physics_engine.constants import MAX_STEAM_FLOW

        return self.calculate(
            steam_temp_in=self.params.nominal_inlet_temp,
            steam_pressure_in=self.params.nominal_inlet_pressure,
            steam_flow=MAX_STEAM_FLOW,
        )
