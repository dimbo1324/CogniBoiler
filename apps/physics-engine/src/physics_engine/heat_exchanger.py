"""
Heat exchanger models for steam boiler auxiliary components.

Models two key heat exchangers absent from the base boiler drum model:

    Superheater:
        Takes saturated steam from the drum and heats it to
        superheated conditions (540–565°C) using flue gas.
        Located in the high-temperature flue gas path.

    Economizer:
        Preheats feedwater before it enters the drum using
        residual heat from flue gas leaving the superheater.
        Recovers waste heat, improving overall boiler efficiency.

Flue gas path (temperature decreasing):
    Furnace → Superheater → Boiler drum tubes → Economizer → Stack

Energy balance for each heat exchanger:
    Q = U * A * LMTD
    where LMTD = log mean temperature difference between hot and cold streams
"""

import math
from dataclasses import dataclass

from physics_engine import steam_tables
from physics_engine.constants import (
    TEMP_STEAM_NOMINAL,
)

# ─── Heat exchanger design constants ─────────────────────────────────────────

# Superheater
SH_HEAT_TRANSFER_AREA: float = 800.0  # m²  — superheater tube surface area
SH_OVERALL_HTC: float = 60.0  # W/(m²·K) — overall heat transfer coefficient
#   (flue gas side dominates, low value)
SH_TUBE_MASS: float = 12000.0  # kg  — thermal mass of superheater tubes
SH_TUBE_CP: float = 500.0  # J/(kg·K) — specific heat of steel tubes

# Economizer
ECO_HEAT_TRANSFER_AREA: float = 1200.0  # m²  — economizer tube surface area
ECO_OVERALL_HTC: float = 45.0  # W/(m²·K) — overall heat transfer coefficient
ECO_TUBE_MASS: float = 8000.0  # kg  — thermal mass of economizer tubes
ECO_TUBE_CP: float = 500.0  # J/(kg·K) — specific heat of steel tubes

# Minimum temperature approach (pinch point) [K]
# Flue gas cannot be cooled below this margin above water/steam temperature
MIN_TEMP_APPROACH: float = 15.0  # K


@dataclass
class SuperheaterState:
    """
    Operating state of the superheater at a given time step.

    Tracks steam conditions entering and leaving the superheater,
    and the heat transferred from flue gas.
    """

    steam_temp_in: float  # K  — saturated steam temperature entering superheater
    steam_temp_out: float  # K  — superheated steam temperature leaving superheater
    flue_gas_temp_in: float  # K  — flue gas temperature entering superheater
    flue_gas_temp_out: float  # K  — flue gas temperature leaving superheater
    heat_transferred: float  # W  — heat transferred from flue gas to steam
    steam_enthalpy_in: float  # J/kg — specific enthalpy of steam entering
    steam_enthalpy_out: float  # J/kg — specific enthalpy of steam leaving


@dataclass
class EconomizerState:
    """
    Operating state of the economizer at a given time step.

    Tracks feedwater conditions and residual flue gas heat recovery.
    """

    water_temp_in: float  # K  — feedwater temperature entering economizer
    water_temp_out: float  # K  — feedwater temperature leaving economizer (→ drum)
    flue_gas_temp_in: float  # K  — flue gas temperature entering economizer
    flue_gas_temp_out: float  # K  — flue gas temperature leaving economizer (stack)
    heat_transferred: float  # W  — heat transferred from flue gas to feedwater
    water_enthalpy_gain: float  # J/kg — enthalpy gain per kg of feedwater


def _lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
) -> float:
    """
    Log Mean Temperature Difference for a counter-flow heat exchanger [K].

    LMTD = (ΔT1 - ΔT2) / ln(ΔT1 / ΔT2)
    where ΔT1 = T_hot_in - T_cold_out
          ΔT2 = T_hot_out - T_cold_in

    Falls back to arithmetic mean if temperatures are nearly equal.
    """
    dt1 = max(t_hot_in - t_cold_out, MIN_TEMP_APPROACH)
    dt2 = max(t_hot_out - t_cold_in, MIN_TEMP_APPROACH)

    if abs(dt1 - dt2) < 0.1:
        return (dt1 + dt2) / 2.0  # arithmetic mean when ΔT1 ≈ ΔT2

    return (dt1 - dt2) / math.log(dt1 / dt2)


class SuperheaterModel:
    """
    Superheater heat exchanger model.

    Takes saturated steam from the boiler drum and superheats it
    using high-temperature flue gas from the furnace.

    The superheater is the hottest heat exchanger in the flue gas path,
    seeing flue gas temperatures of 800–1100°C at full load.

    Key output: steam_temp_out — this is what drives turbine efficiency.
    Target: 540–565°C (813–838 K).
    """

    def __init__(
        self,
        area: float = SH_HEAT_TRANSFER_AREA,
        htc: float = SH_OVERALL_HTC,
    ) -> None:
        self.area = area
        self.htc = htc
        self.ua = area * htc  # W/K — overall conductance

    def calculate(
        self,
        pressure_pa: float,
        steam_flow: float,
        flue_gas_temp_in: float,
        flue_gas_flow: float,
        cp_flue_gas: float = 1100.0,
    ) -> SuperheaterState:
        """
        Calculate superheater performance at given operating conditions.

        Uses effectiveness-NTU method for counter-flow heat exchanger.

        Args:
            pressure_pa: Steam drum pressure [Pa].
            steam_flow: Steam mass flow through superheater [kg/s].
            flue_gas_temp_in: Flue gas temperature entering superheater [K].
            flue_gas_flow: Flue gas mass flow [kg/s].
            cp_flue_gas: Specific heat of flue gas [J/(kg·K)].

        Returns:
            SuperheaterState with all calculated temperatures and heat transfer.
        """
        # ── Saturated steam conditions at drum pressure ───────────────────────
        t_sat = steam_tables.saturation_temp(pressure_pa)
        h_steam_in = steam_tables.saturated_vapor_enthalpy(pressure_pa)

        # ── Capacity rates [W/K] ──────────────────────────────────────────────
        # C_steam: use average Cp between saturation and target superheat
        cp_steam_avg = steam_tables.steam_specific_heat(
            temp_k=(t_sat + TEMP_STEAM_NOMINAL) / 2.0,
            pressure_pa=pressure_pa,
        )
        c_steam = steam_flow * cp_steam_avg if steam_flow > 0.0 else 1.0
        c_gas = flue_gas_flow * cp_flue_gas if flue_gas_flow > 0.0 else 1.0

        c_min = min(c_steam, c_gas)
        c_max = max(c_steam, c_gas)
        c_ratio = c_min / c_max

        # ── Effectiveness-NTU method ──────────────────────────────────────────
        ntu = self.ua / c_min
        if c_ratio < 0.99:
            effectiveness = (1.0 - math.exp(-ntu * (1.0 - c_ratio))) / (
                1.0 - c_ratio * math.exp(-ntu * (1.0 - c_ratio))
            )
        else:
            effectiveness = ntu / (1.0 + ntu)  # balanced flow case

        # ── Maximum possible heat transfer ────────────────────────────────────
        q_max = c_min * (flue_gas_temp_in - t_sat)
        q_actual = effectiveness * q_max
        q_actual = max(0.0, q_actual)

        # ── Outlet temperatures ───────────────────────────────────────────────
        t_steam_out = t_sat + q_actual / c_steam
        t_gas_out = flue_gas_temp_in - q_actual / c_gas

        # Enforce pinch: flue gas cannot be cooled below steam temp + margin
        t_gas_out = max(t_gas_out, t_steam_out + MIN_TEMP_APPROACH)

        # ── Steam outlet enthalpy ─────────────────────────────────────────────
        h_steam_out = steam_tables.steam_enthalpy(
            temp_k=t_steam_out,
            pressure_pa=pressure_pa,
        )

        return SuperheaterState(
            steam_temp_in=t_sat,
            steam_temp_out=t_steam_out,
            flue_gas_temp_in=flue_gas_temp_in,
            flue_gas_temp_out=t_gas_out,
            heat_transferred=q_actual,
            steam_enthalpy_in=h_steam_in,
            steam_enthalpy_out=h_steam_out,
        )


class EconomizerModel:
    """
    Economizer (feedwater preheater) heat exchanger model.

    Recovers residual heat from flue gas leaving the superheater
    and uses it to preheat feedwater before it enters the drum.

    Benefits:
        1. Reduces heat input needed to bring feedwater to saturation
        2. Lowers stack temperature → improves overall boiler efficiency
        3. Reduces thermal shock to drum (water enters closer to saturation)

    Typical feedwater temperature rise: 150°C → 230°C (423 K → 503 K)
    Typical flue gas temperature drop: 400°C → 180°C (673 K → 453 K)
    """

    def __init__(
        self,
        area: float = ECO_HEAT_TRANSFER_AREA,
        htc: float = ECO_OVERALL_HTC,
    ) -> None:
        self.area = area
        self.htc = htc
        self.ua = area * htc  # W/K

    def calculate(
        self,
        feedwater_flow: float,
        feedwater_temp_in: float,
        pressure_pa: float,
        flue_gas_temp_in: float,
        flue_gas_flow: float,
        cp_flue_gas: float = 1100.0,
    ) -> EconomizerState:
        """
        Calculate economizer performance at given operating conditions.

        Args:
            feedwater_flow: Feedwater mass flow [kg/s].
            feedwater_temp_in: Feedwater inlet temperature [K].
            pressure_pa: Drum pressure [Pa] — used to find saturation temp
                         (feedwater must not boil in economizer).
            flue_gas_temp_in: Flue gas temperature entering economizer [K].
            flue_gas_flow: Flue gas mass flow [kg/s].
            cp_flue_gas: Specific heat of flue gas [J/(kg·K)].

        Returns:
            EconomizerState with all calculated temperatures and heat recovery.
        """
        # ── Saturation temperature limit ──────────────────────────────────────
        # Feedwater must not reach saturation inside the economizer
        # (would cause steam flashing — dangerous and inefficient)
        t_sat = steam_tables.saturation_temp(pressure_pa)
        t_water_max = t_sat - 10.0  # K — 10 K subcooling margin

        # ── Water specific heat at average conditions ─────────────────────────
        t_water_avg = (feedwater_temp_in + t_water_max) / 2.0
        cp_water = steam_tables.water_specific_heat(
            temp_k=t_water_avg,
            pressure_pa=pressure_pa,
        )

        # ── Capacity rates [W/K] ──────────────────────────────────────────────
        c_water = feedwater_flow * cp_water if feedwater_flow > 0.0 else 1.0
        c_gas = flue_gas_flow * cp_flue_gas if flue_gas_flow > 0.0 else 1.0

        c_min = min(c_water, c_gas)
        c_max = max(c_water, c_gas)
        c_ratio = c_min / c_max

        # ── Effectiveness-NTU ─────────────────────────────────────────────────
        ntu = self.ua / c_min
        if c_ratio < 0.99:
            effectiveness = (1.0 - math.exp(-ntu * (1.0 - c_ratio))) / (
                1.0 - c_ratio * math.exp(-ntu * (1.0 - c_ratio))
            )
        else:
            effectiveness = ntu / (1.0 + ntu)

        # ── Heat transfer ─────────────────────────────────────────────────────
        q_max = c_min * (flue_gas_temp_in - feedwater_temp_in)
        q_actual = effectiveness * q_max
        q_actual = max(0.0, q_actual)

        # ── Outlet temperatures ───────────────────────────────────────────────
        t_water_out = feedwater_temp_in + q_actual / c_water
        t_water_out = min(t_water_out, t_water_max)  # enforce subcooling limit
        t_gas_out = flue_gas_temp_in - q_actual / c_gas
        t_gas_out = max(t_gas_out, feedwater_temp_in + MIN_TEMP_APPROACH)

        # ── Feedwater enthalpy gain ───────────────────────────────────────────
        h_in = steam_tables.water_enthalpy(feedwater_temp_in, pressure_pa)
        h_out = steam_tables.water_enthalpy(t_water_out, pressure_pa)
        enthalpy_gain = h_out - h_in

        return EconomizerState(
            water_temp_in=feedwater_temp_in,
            water_temp_out=t_water_out,
            flue_gas_temp_in=flue_gas_temp_in,
            flue_gas_temp_out=t_gas_out,
            heat_transferred=q_actual,
            water_enthalpy_gain=enthalpy_gain,
        )
