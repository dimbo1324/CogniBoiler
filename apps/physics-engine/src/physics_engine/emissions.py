"""
Emissions model for natural gas combustion.

Calculates real-time stack emission rates:
    CO2  — stoichiometric combustion product, proportional to fuel flow
    NOx  — thermal NOx via Zeldovich mechanism; exponential of flame temperature
    CO   — incomplete combustion product; rises at low excess air or high loads

All rates in kg/s at the stack.

Portfolio note: emissions are a critical KPI for modern power plants.
CO2 intensity (kg/MWh) directly drives carbon credit costs, and NOx limits
are enforced by environmental regulators (EU IED Directive, EPA 40 CFR).

References:
    - Borman & Ragland (1998), "Combustion Engineering", McGraw-Hill
    - European Environment Agency, EMEP/EEA Guidebook 2019, Chapter 1.A.1
    - US EPA AP-42, Section 1.4: Natural Gas Combustion
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.constants import FUEL_CO2_YIELD, FUEL_HEATING_VALUE

# ─── Physical constants ───────────────────────────────────────────────────────

# CO2 follows the carbon in the fuel as fired (see constants.FUEL_CO2_YIELD):
# 2.343 kg CO2 / kg fuel, that is 55.8 g CO2 per MJ — the figure of pipeline natural gas.
CO2_EMISSION_FACTOR: float = FUEL_CO2_YIELD  # kg CO2 / kg fuel

# ── Thermal NOx (Zeldovich mechanism) ─────────────────────────────────────────
#
# The extended Zeldovich mechanism gives:
#   d[NO]/dt = 2·k1·[O]·[N2]  (rate-limiting step)
#
# k1 = 1.8e14 · exp(−38 370 / T), so the emission index follows an activation
# *temperature*, not an activation energy divided by R:
#   EI_NOx [kg/kg_fuel] = A · exp(−T_a / T_flame)
#
# A is calibrated to a low-NOx gas burner: a 1 700 K flame zone gives 1.5 g NOx per kg
# of fuel (≈ 45 ppmv). The same law gives ≈ 0.07 g/kg at 1 500 K and ≈ 16 g/kg at 1 900 K.
# The caller passes the flame-zone temperature, not the adiabatic flame temperature: the
# plant feeds the furnace exit gas plus NOX_FLAME_ZONE_OFFSET_K, which is 1 700 K at rated
# load, so fouling or a leaner mixture moves NOx the way it does on a real unit.
NOX_ACTIVATION_TEMPERATURE: float = 38_370.0  # K
NOX_CALIBRATION_TEMP: float = 1_700.0  # K — flame zone at rated load
NOX_CALIBRATION_EI: float = 0.0015  # kg NOx / kg fuel at the calibration temperature
NOX_PRE_EXP: float = NOX_CALIBRATION_EI / math.exp(
    -NOX_ACTIVATION_TEMPERATURE / NOX_CALIBRATION_TEMP
)

# ── CO from incomplete combustion ─────────────────────────────────────────────
CO_BASE_EI: float = 0.0003  # kg CO / kg fuel at λ = 1.10 (design point)
CO_RICH_PENALTY: float = 0.050  # additional kg CO / kg fuel at λ = 1.00


# ─── Emissions state ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmissionsState:
    """
    Instantaneous emission rates at the stack.

    All rates in kg/s; multiply by 3600 to get kg/h.
    Divide by electrical_power_mw to get specific emissions (kg/MWh).

    Attributes:
        co2_rate:   CO2 emission rate [kg/s]
        nox_rate:   NOx emission rate [kg/s] (as NO2 equivalent)
        co_rate:    CO emission rate  [kg/s]
        fuel_flow:  Fuel mass flow    [kg/s]  — for downstream normalisation
        nox_ppmv:   NOx dry-stack concentration [ppmv, 3 % O2 reference]
    """

    co2_rate: float  # kg/s
    nox_rate: float  # kg/s
    co_rate: float  # kg/s
    fuel_flow: float  # kg/s
    nox_ppmv: float  # ppmv (dry, 3 % O2)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def co2_rate_kg_per_h(self) -> float:
        """CO2 emission rate [kg/h]."""
        return self.co2_rate * 3600.0

    @property
    def nox_rate_mg_per_s(self) -> float:
        """NOx emission rate [mg/s]."""
        return self.nox_rate * 1.0e6

    @property
    def co2_intensity_kg_per_mwh(self) -> float:
        """
        Specific CO2 intensity [kg CO2 / MWh of fuel input].

        Fuel-specific value (independent of electrical output): the electrical intensity
        is this divided by the net efficiency, and PerformanceMsg carries that one.
        Natural gas is ~200 kg per MWh of fuel on an LHV basis; coal ~340.
        """
        if self.fuel_flow <= 0.0:
            return 0.0
        fuel_power_mw = self.fuel_flow * FUEL_HEATING_VALUE / 1.0e6
        return (self.co2_rate * 3600.0) / fuel_power_mw


# ─── Emissions calculator ─────────────────────────────────────────────────────


class EmissionsCalculator:
    """
    Calculates real-time stack emissions from combustion conditions.

    Designed to be called at every simulation timestep from the ScenarioRunner.
    Stateless — all computation is in calculate().

    Usage:
        calc = EmissionsCalculator()
        state = calc.calculate(
            fuel_flow=5.0,         # kg/s
            flame_temp=1 650.0,    # K
            excess_air_ratio=1.10,
        )
        print(f"CO2: {state.co2_rate:.2f} kg/s  "
              f"NOx: {state.nox_ppmv:.0f} ppmv  "
              f"CO: {state.co_rate*1e6:.1f} mg/s")
    """

    def calculate(
        self,
        fuel_flow: float,
        flame_temp: float,
        excess_air_ratio: float,
    ) -> EmissionsState:
        """
        Calculate emission rates for given combustion conditions.

        Args:
            fuel_flow:        Fuel mass flow [kg/s].
            flame_temp:       Adiabatic flame temperature [K].
                              Typical: 1 400–2 000 K.
            excess_air_ratio: Lambda (λ). 1.0 = stoichiometric.
                              Design point: 1.05–1.15 for gas burners.

        Returns:
            EmissionsState with CO2, NOx, CO rates and NOx in ppmv.
        """
        fuel_flow = max(0.0, fuel_flow)
        flame_temp = max(300.0, flame_temp)
        lam = max(0.5, min(excess_air_ratio, 3.0))

        # ── CO2: exact stoichiometry, independent of flame conditions ─────────
        co2_rate = fuel_flow * CO2_EMISSION_FACTOR

        # ── Thermal NOx via Zeldovich ──────────────────────────────────────────
        # EI [kg/kg_fuel] = NOX_PRE_EXP · exp(−T_a / T_flame)
        t_clamped = max(1000.0, min(flame_temp, 2200.0))
        ei_nox_base = NOX_PRE_EXP * math.exp(-NOX_ACTIVATION_TEMPERATURE / t_clamped)

        # Excess air correction:
        #   Fuel-rich (λ < 1):  less O2 available → lower NOx
        #   Near-optimal (1.0–1.15): peak NOx window
        #   Excess air (λ > 1.15): lower flame T due to dilution → less NOx
        if lam < 1.0:
            nox_lambda_factor = lam  # linear drop
        elif lam <= 1.15:
            nox_lambda_factor = 1.0  # optimal zone
        else:
            nox_lambda_factor = max(0.4, 1.0 - 0.4 * (lam - 1.15))  # dilution

        nox_rate = fuel_flow * ei_nox_base * nox_lambda_factor

        # ── CO: incomplete combustion ──────────────────────────────────────────
        if lam >= 1.05:
            co_ei = CO_BASE_EI
        elif lam >= 1.0:
            # Rises sharply between λ=1.00 and λ=1.05
            alpha = (1.05 - lam) / 0.05
            co_ei = CO_BASE_EI + alpha * CO_RICH_PENALTY
        else:
            # Fuel-rich: large CO spike (incomplete combustion)
            co_ei = CO_BASE_EI + CO_RICH_PENALTY * (1.0 + 4.0 * (1.0 - lam))

        co_rate = fuel_flow * co_ei

        # ── NOx in ppmv (dry, 3 % O2 reference) ───────────────────────────────
        # Empirical conversion: 1 g NOx / kg fuel ≈ 30 ppmv at 3 % O2 reference
        # (based on stoichiometry of ~10 m³ flue gas per kg natural gas)
        nox_ei_g_per_kg = ei_nox_base * nox_lambda_factor * 1000.0
        nox_ppmv = nox_ei_g_per_kg * 30.0

        return EmissionsState(
            co2_rate=co2_rate,
            nox_rate=nox_rate,
            co_rate=co_rate,
            fuel_flow=fuel_flow,
            nox_ppmv=nox_ppmv,
        )
