"""
Surface condenser and condensate return cycle model.

Models the closed steam-water cycle from turbine exhaust back to drum feedwater:

    Turbine exhaust → Surface condenser → Condensate pump
        → Low-pressure feedwater heaters → Deaerator → Feed pump → Drum

Key physical effects added by this model vs fixed TEMP_FEEDWATER constant:

    1. Dynamic feedwater temperature: varies with steam load and cooling water.
       At low load → better condenser performance → colder feedwater → lower drum temp.
       At high load → higher backpressure → hotter condensate.

    2. Dynamic turbine backpressure: increases at high steam flow (more heat to reject).
       Higher backpressure → less turbine enthalpy drop → lower electrical output.
       Effect: ~0.3–0.5% power loss per 1 kPa backpressure increase.

    3. Seasonal variation: cooling water temperature (river/sea) affects cycle efficiency.
       A 5°C rise in cooling water temp → ~0.5% drop in net cycle efficiency.

Physical basis: 300 MW steam unit, surface condenser, two-pass shell-and-tube.
Design point: backpressure 7 kPa, CW inlet 15°C, CW outlet 27°C.

References:
    - El-Wakil (1984), "Power Plant Technology", McGraw-Hill, Chapter 11
    - EPRI CS-3243: Performance Testing of Condensers
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine import properties

# ─── Design parameters ────────────────────────────────────────────────────────

# Overall UA for condensing heat transfer (W/K)
# A 300 MW unit rejects ~550 MW. Holding the 7 kPa design back-pressure (39 °C
# condensing) with 15 °C cooling water at 8 000 kg/s needs an effectiveness of ~0.7,
# i.e. NTU ≈ 1.2 → UA ≈ 40 MW/K (~13 000 m² at 3 000 W/(m²·K)). 12 MW/K saturated the
# condenser at its 25 kPa limit and cost the turbine a tenth of its work.
CONDENSER_UA: float = 40.0e6  # W/K

COOLING_WATER_FLOW: float = 8_000.0  # kg/s — typical for 300 MW unit
COOLING_WATER_TEMP_DESIGN: float = 288.15  # K  — 15°C design CW inlet

# Deaerator: direct-contact heater, removes dissolved O2 from condensate.
# Operates at fixed pressure; condensate exits at saturation temperature.
DEAERATOR_PRESSURE: float = 3.5e5  # Pa — 3.5 bar
DEAERATOR_TEMP: float = 412.15  # K  — saturation temp at 3.5 bar ≈ 139°C

# Condenser pressure limits [Pa]
MIN_BACKPRESSURE: float = 3_000.0  # Pa — 0.03 bar (excellent vacuum)
MAX_BACKPRESSURE: float = 25_000.0  # Pa — 0.25 bar (dirty condenser / high CW temp)
DESIGN_BACKPRESSURE: float = 7_000.0  # Pa — 0.07 bar (design point)

# Reference heat duty at full load (300 MW electrical, ~30% efficiency loss)
DESIGN_HEAT_DUTY: float = 700.0e6  # W — 700 MW rejected to cooling water

# Specific heat of water [J/(kg·K)]
CP_WATER: float = 4_186.0


# ─── Condenser state ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CondenserState:
    """
    Thermodynamic state of the surface condenser and condensate return system.

    Attributes:
        backpressure_pa:        Condenser pressure = turbine exhaust pressure [Pa].
        condensate_temp:        Hotwell condensate temperature [K].
        feedwater_temp:         Feedwater temperature after deaerator [K].
                                This replaces the fixed TEMP_FEEDWATER constant.
        cooling_water_temp_out: Cooling water outlet temperature [K].
        heat_rejected_w:        Total heat transferred to cooling water [W].
        condenser_loading:      Fraction of design heat duty [-].
    """

    backpressure_pa: float  # Pa
    condensate_temp: float  # K
    feedwater_temp: float  # K
    cooling_water_temp_out: float  # K
    heat_rejected_w: float  # W
    condenser_loading: float  # [0, 1]

    @property
    def backpressure_bar(self) -> float:
        """Condenser backpressure [bar]."""
        return self.backpressure_pa / 1.0e5

    @property
    def backpressure_kpa(self) -> float:
        """Condenser backpressure [kPa]."""
        return self.backpressure_pa / 1_000.0

    @property
    def feedwater_temp_celsius(self) -> float:
        """Feedwater temperature [°C]."""
        return self.feedwater_temp - 273.15

    @property
    def cycle_efficiency_loss_pct(self) -> float:
        """
        Approximate efficiency loss vs design backpressure [%].

        Rule of thumb: +1 kPa above design → ~0.3% efficiency drop.
        Used as a KPI on the Grafana efficiency dashboard.
        """
        delta_kpa = max(0.0, (self.backpressure_pa - DESIGN_BACKPRESSURE) / 1_000.0)
        return delta_kpa * 0.3


# ─── Condenser model ──────────────────────────────────────────────────────────


class CondenserModel:
    """
    Surface condenser model using NTU-effectiveness method.

    At each timestep, given steam flow and inlet conditions, calculates:
      1. Condensing temperature (and hence backpressure)
      2. Cooling water outlet temperature
      3. Feedwater temperature after deaerator

    The condensing side is treated as a constant-temperature fluid
    (isothermal condensation), so effectiveness = 1 − exp(−NTU).

    Fouling increases thermal resistance, raising backpressure:
        UA_fouled = 1 / (1/UA_clean + R_fouling)
    where R_fouling [m²·K/W] grows slowly over time (see EquipmentHealth).

    Usage:
        model = CondenserModel()
        state = model.calculate(
            steam_flow=200.0,           # kg/s
            steam_enthalpy_in=2_350e3,  # J/kg — wet steam at ~0.9 quality
            cooling_water_temp=288.15,  # K — river water temperature
        )
        # Use state.backpressure_pa as turbine exhaust pressure
        # Use state.feedwater_temp as economizer inlet temperature
    """

    def __init__(
        self,
        ua: float = CONDENSER_UA,
        cooling_water_flow: float = COOLING_WATER_FLOW,
        cooling_water_temp: float = COOLING_WATER_TEMP_DESIGN,
        fouling_resistance: float = 0.0,
    ) -> None:
        """
        Args:
            ua:                   Clean-tube overall heat transfer coefficient × area [W/K].
            cooling_water_flow:   Cooling water mass flow [kg/s].
            cooling_water_temp:   Cooling water inlet temperature [K].
            fouling_resistance:   Additional fouling resistance [m²·K/W].
                                  Typical clean: 0.  Fouled after 1 year: ~0.0001.
        """
        self.cooling_water_flow = cooling_water_flow
        self.cooling_water_temp_in = cooling_water_temp

        # Apply fouling: UA_eff = 1 / (1/UA_clean + R_fouling)
        # Fouling resistance is not per-unit-area here — it acts as a global
        # additional resistance scaled to UA magnitude.
        if fouling_resistance > 0.0 and ua > 0.0:
            # Convert: if R_fouling is 0.0001 m²·K/W and surface is ~4000 m²
            # → effective extra resistance = R / A_approx = 0.0001 / (ua / 3000) W/K
            # Simplified direct form: UA_eff = 1 / (1/ua + fouling_resistance / 1000)
            self.ua = 1.0 / (1.0 / ua + fouling_resistance / 1_000.0)
        else:
            self.ua = ua

    def calculate(
        self,
        steam_flow: float,
        steam_enthalpy_in: float,
        cooling_water_temp: float | None = None,
    ) -> CondenserState:
        """
        Calculate condenser operating state at given steam load.

        Args:
            steam_flow:          Steam mass flow entering condenser [kg/s].
            steam_enthalpy_in:   Specific enthalpy of exhaust steam [J/kg].
                                 Typically 2 200–2 500 kJ/kg (wet/saturated).
            cooling_water_temp:  Cooling water inlet temperature [K].
                                 If None, uses design value set in __init__.

        Returns:
            CondenserState with backpressure, condensate and feedwater temperatures.
        """
        steam_flow = max(0.0, steam_flow)
        t_cw_in = (
            cooling_water_temp
            if cooling_water_temp is not None
            else self.cooling_water_temp_in
        )

        # Idle condenser — return design minimum conditions
        if steam_flow < 1.0:
            t_cond = properties.saturation_temperature(MIN_BACKPRESSURE)
            return CondenserState(
                backpressure_pa=MIN_BACKPRESSURE,
                condensate_temp=t_cond,
                feedwater_temp=DEAERATOR_TEMP,
                cooling_water_temp_out=t_cw_in + 1.0,
                heat_rejected_w=0.0,
                condenser_loading=0.0,
            )

        # ── NTU-effectiveness (condensing approximation) ──────────────────────
        # For a condensing fluid: C_steam → ∞, so:
        #   effectiveness = 1 − exp(−NTU)
        #   NTU = UA / C_cw
        c_cw = self.cooling_water_flow * CP_WATER  # W/K
        ntu = self.ua / c_cw
        effectiveness = 1.0 - math.exp(-ntu)

        # ── Find condensing temperature via energy balance iteration ──────────
        # At steady state:
        #   Q_steam = m_steam · (h_steam_in − h_liq(T_cond))
        #   Q_cw    = C_cw · ε · (T_cond − T_cw_in)
        # Solve Q_steam = Q_cw for T_cond using bisection.

        t_cond = self._find_condensing_temp(
            steam_flow=steam_flow,
            h_steam_in=steam_enthalpy_in,
            t_cw_in=t_cw_in,
            c_cw=c_cw,
            effectiveness=effectiveness,
        )

        # ── Derive backpressure from condensing temperature ───────────────────
        backpressure = properties.saturation_pressure(t_cond)
        backpressure = max(MIN_BACKPRESSURE, min(backpressure, MAX_BACKPRESSURE))

        # ── Re-compute heat duty at final backpressure ────────────────────────
        h_liq = properties.liquid_enthalpy_at_pressure(backpressure)
        q_rejected = steam_flow * max(0.0, steam_enthalpy_in - h_liq)

        # Cooling water outlet temperature
        t_cw_out = t_cw_in + q_rejected / (self.cooling_water_flow * CP_WATER)

        # ── Condenser loading ─────────────────────────────────────────────────
        loading = min(1.0, q_rejected / DESIGN_HEAT_DUTY)

        return CondenserState(
            backpressure_pa=backpressure,
            condensate_temp=t_cond,
            feedwater_temp=DEAERATOR_TEMP,  # deaerator fixes feedwater temp
            cooling_water_temp_out=t_cw_out,
            heat_rejected_w=q_rejected,
            condenser_loading=loading,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_condensing_temp(
        self,
        steam_flow: float,
        h_steam_in: float,
        t_cw_in: float,
        c_cw: float,
        effectiveness: float,
    ) -> float:
        """
        Find condensing temperature where Q_steam = Q_cooling_water.

        Uses bisection between (T_cw_in + 3 K) and 75°C max.
        Typically converges in < 20 iterations to < 0.1 K.
        """
        t_lo = t_cw_in + 3.0  # K — minimum: 3 K above cooling water
        t_hi = 348.15  # K — 75°C maximum for vacuum condenser

        for _ in range(40):
            t_mid = (t_lo + t_hi) / 2.0

            # Steam-side heat: condensation from h_in to saturated liquid at T_mid
            h_liq_mid = properties.liquid_enthalpy(t_mid)
            q_steam = steam_flow * max(0.0, h_steam_in - h_liq_mid)

            # Cooling water side
            q_cw = c_cw * effectiveness * (t_mid - t_cw_in)

            if abs(q_steam - q_cw) < 500.0:  # W — 0.5 kW convergence
                break
            if q_steam > q_cw:
                t_lo = t_mid  # higher T needed to reject more heat to CW
            else:
                t_hi = t_mid

        return (t_lo + t_hi) / 2.0
