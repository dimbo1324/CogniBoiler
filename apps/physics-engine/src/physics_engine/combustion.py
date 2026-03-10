"""
Combustion model for natural gas firing in a steam boiler furnace.

Models the heat release from fuel combustion accounting for:
    - Fuel composition (methane-dominated natural gas)
    - Excess air ratio (lambda)
    - Inlet air temperature (preheating effect)
    - Combustion efficiency as a function of excess air

Physics:
    Q_released = m_fuel * LHV * eta_combustion(lambda)
    Q_available = Q_released + Q_air_preheat - Q_flue_gas_loss
"""

from dataclasses import dataclass

from physics_engine.constants import (
    COMBUSTION_EFFICIENCY,
    FUEL_HEATING_VALUE,
    MAX_FUEL_FLOW,
    TEMP_AMBIENT,
)

# ─── Natural gas composition (mole fractions) ─────────────────────────────────
# Typical pipeline natural gas composition
METHANE_FRACTION: float = 0.92  # CH4
ETHANE_FRACTION: float = 0.04  # C2H6
PROPANE_FRACTION: float = 0.02  # C3H8
NITROGEN_FRACTION: float = 0.02  # N2 (inert)

# Stoichiometric air-fuel ratio for natural gas [kg_air / kg_fuel]
STOICHIOMETRIC_AFR: float = 17.2

# Specific heat of combustion air [J/(kg·K)]
CP_AIR: float = 1010.0

# Specific heat of flue gas [J/(kg·K)]
CP_FLUE_GAS: float = 1100.0

# Flue gas mass per kg of fuel at stoichiometric conditions [kg_flue/kg_fuel]
FLUE_GAS_PER_FUEL: float = 18.2

# Reference lower heating value of natural gas [J/kg]
LHV_NATURAL_GAS: float = FUEL_HEATING_VALUE  # 42.0 MJ/kg


@dataclass
class CombustionState:
    """
    Result of combustion calculation for a single time step.

    All power values in Watts [W], temperatures in Kelvin [K],
    mass flows in kg/s.
    """

    fuel_flow: float  # kg/s  — actual fuel mass flow
    air_flow: float  # kg/s  — actual combustion air mass flow
    flue_gas_flow: float  # kg/s  — flue gas mass flow at furnace exit
    heat_released: float  # W     — total heat from combustion
    heat_available: float  # W     — net heat available to boiler
    flue_gas_temp_exit: float  # K     — adiabatic flame temperature estimate
    excess_air_ratio: float  # —     — lambda (1.0 = stoichiometric)
    eta_combustion: float  # —     — actual combustion efficiency


class CombustionModel:
    """
    Combustion model for natural gas burner in a steam boiler.

    Calculates heat release, air requirements, and flue gas properties
    as a function of fuel flow and operating conditions.

    Key concept — excess air ratio (lambda):
        lambda = 1.0 : perfect stoichiometric combustion (theoretical)
        lambda < 1.0 : fuel-rich, incomplete combustion, CO formation
        lambda > 1.0 : excess air, more complete combustion but more heat lost
        Optimal: lambda = 1.05 to 1.15 for natural gas boilers
    """

    def __init__(
        self,
        max_fuel_flow: float = MAX_FUEL_FLOW,
        nominal_excess_air: float = 1.1,
        air_preheat_temp: float = TEMP_AMBIENT,
    ) -> None:
        """
        Args:
            max_fuel_flow: Maximum fuel mass flow rate [kg/s].
            nominal_excess_air: Design excess air ratio (lambda). Typical: 1.05–1.15.
            air_preheat_temp: Combustion air inlet temperature [K].
                              Higher = better efficiency (air preheater effect).
        """
        self.max_fuel_flow = max_fuel_flow
        self.nominal_excess_air = nominal_excess_air
        self.air_preheat_temp = air_preheat_temp

    def _combustion_efficiency(self, excess_air_ratio: float) -> float:
        """
        Calculate combustion efficiency as a function of excess air.

        Efficiency peaks near lambda=1.05 and drops for both rich and lean
        mixtures. Models incomplete combustion (rich) and excessive flue
        gas losses (lean).

        Args:
            excess_air_ratio: Lambda value.

        Returns:
            Combustion efficiency [0, 1].
        """
        if excess_air_ratio < 1.0:
            # Fuel-rich: incomplete combustion, CO formation
            # Efficiency drops rapidly below stoichiometric
            eta = COMBUSTION_EFFICIENCY * (0.5 + 0.5 * excess_air_ratio)
        elif excess_air_ratio <= 1.05:
            # Optimal zone: near-perfect combustion
            eta = COMBUSTION_EFFICIENCY
        else:
            # Excess air: more complete combustion but heat lost to extra flue gas
            # Each 10% excess air reduces efficiency ~0.5%
            excess = excess_air_ratio - 1.05
            eta = COMBUSTION_EFFICIENCY - 0.05 * excess
        return max(0.0, min(1.0, eta))

    def _adiabatic_flame_temp(
        self,
        fuel_flow: float,
        air_flow: float,
        heat_released: float,
        flue_gas_flow: float,
    ) -> float:
        """
        Estimate adiabatic flame temperature [K].

        Energy balance: Q_released = m_flue * Cp_flue * (T_flame - T_air)
        This is the maximum possible furnace temperature.
        """
        if flue_gas_flow <= 0.0:
            return TEMP_AMBIENT

        delta_t = heat_released / (flue_gas_flow * CP_FLUE_GAS)
        return self.air_preheat_temp + delta_t

    def calculate(
        self,
        fuel_valve: float,
        excess_air_ratio: float | None = None,
    ) -> CombustionState:
        """
        Calculate combustion state for given fuel valve position.

        Args:
            fuel_valve: Fuel valve position [0, 1].
            excess_air_ratio: Lambda override. If None, uses nominal value.

        Returns:
            CombustionState with all calculated quantities.
        """
        fuel_valve = max(0.0, min(1.0, fuel_valve))
        lam = (
            excess_air_ratio
            if excess_air_ratio is not None
            else self.nominal_excess_air
        )
        lam = max(0.5, min(lam, 2.0))  # physical limits

        # ── Fuel and air flows ────────────────────────────────────────────────
        m_fuel = fuel_valve * self.max_fuel_flow
        m_air = m_fuel * STOICHIOMETRIC_AFR * lam
        m_flue = m_fuel + m_air
        # ── Combustion efficiency ─────────────────────────────────────────────
        eta = self._combustion_efficiency(lam)

        # ── Heat release ──────────────────────────────────────────────────────
        q_released = m_fuel * LHV_NATURAL_GAS * eta

        # ── Air preheat contribution ──────────────────────────────────────────
        # Hot combustion air brings additional sensible heat into furnace
        q_air_preheat = m_air * CP_AIR * (self.air_preheat_temp - TEMP_AMBIENT)

        # ── Total heat available to furnace ───────────────────────────────────
        q_available = q_released + q_air_preheat

        # ── Adiabatic flame temperature ───────────────────────────────────────
        t_flame = self._adiabatic_flame_temp(m_fuel, m_air, q_released, m_flue)

        return CombustionState(
            fuel_flow=m_fuel,
            air_flow=m_air,
            flue_gas_flow=m_flue,
            heat_released=q_released,
            heat_available=q_available,
            flue_gas_temp_exit=t_flame,
            excess_air_ratio=lam,
            eta_combustion=eta,
        )

    def flue_gas_heat_loss(
        self, flue_gas_temp_exit: float, flue_gas_flow: float
    ) -> float:
        """
        Heat lost with exiting flue gas through the stack [W].

        This is the main boiler efficiency loss. Lower exit temp = better efficiency.
        Typical stack temperature target: 120–160°C (393–433 K).

        Args:
            flue_gas_temp_exit: Flue gas temperature leaving the boiler [K].
            flue_gas_flow: Flue gas mass flow [kg/s].

        Returns:
            Heat loss rate [W].
        """
        target_stack_temp = 423.15  # K — 150°C target stack temperature
        delta_t = max(flue_gas_temp_exit - target_stack_temp, 0.0)
        return flue_gas_flow * CP_FLUE_GAS * delta_t
