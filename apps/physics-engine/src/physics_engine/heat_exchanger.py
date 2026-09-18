"""
Convective heat exchangers in the flue gas path of the drum boiler.

Flue gas leaves the furnace and crosses, in order of falling temperature:

    Superheater:
        Heats saturated steam from the drum to superheated conditions.

    Evaporator bank:
        Convective boiler tubes that raise more steam from drum water.

    Economizer:
        Preheats feedwater before it enters the drum, recovering residual heat
        before the gas reaches the stack.

Heat transfer method: effectiveness-NTU. Every exchanger conserves energy exactly: the
gas loses what the water or steam gains. The design surfaces come from the rated heat
balance of the 300 MW unit (furnace exit 1 400 K, bank exit ~690 K, stack ~443 K).
"""

import math
from dataclasses import dataclass

from physics_engine import properties
from physics_engine.constants import FLUE_GAS_CP

# ─── Heat exchanger design constants ─────────────────────────────────────────

# Superheater: ~213 MW at rated load, steam 610 K -> ~855 K before spray.
SH_HEAT_TRANSFER_AREA: float = 4_000.0  # m²
SH_OVERALL_HTC: float = 130.0  # W/(m²·K)

# Evaporator bank: ~90 MW at rated load, gas ~900 K -> ~690 K.
BANK_HEAT_TRANSFER_AREA: float = 6_000.0  # m²
BANK_OVERALL_HTC: float = 91.0  # W/(m²·K)

# Economizer: ~105 MW at rated load, feedwater 423 K -> ~522 K (finned tubes).
ECO_HEAT_TRANSFER_AREA: float = 25_000.0  # m²
ECO_OVERALL_HTC: float = 60.0  # W/(m²·K)

# Feedwater leaving the economizer stays this far below saturation (no steaming).
ECO_SUBCOOLING_MARGIN: float = 10.0  # K

CP_FLUE_GAS: float = FLUE_GAS_CP  # J/(kg·K)


def counterflow_effectiveness(ua: float, c_a: float, c_b: float) -> float:
    """Effectiveness of a counterflow exchanger with conductance `ua` [W/K]."""
    c_min = min(c_a, c_b)
    c_max = max(c_a, c_b)
    if c_min <= 0.0:
        return 0.0
    ntu = ua / c_min
    ratio = c_min / c_max
    if ratio > 0.999:
        return ntu / (1.0 + ntu)
    decay = math.exp(-ntu * (1.0 - ratio))
    return (1.0 - decay) / (1.0 - ratio * decay)


@dataclass
class SuperheaterState:
    """
    Operating state of the superheater at a given time step.

    Tracks steam conditions entering and leaving the superheater,
    and the heat transferred from flue gas.
    """

    steam_temp_in: float  # K    — saturated steam temperature entering
    steam_temp_out: float  # K   — superheated steam temperature leaving
    flue_gas_temp_in: float  # K  — flue gas temperature entering superheater
    flue_gas_temp_out: float  # K — flue gas temperature leaving superheater
    heat_transferred: float  # W  — heat transferred from flue gas to steam
    steam_enthalpy_in: float  # J/kg — specific enthalpy of steam entering
    steam_enthalpy_out: float  # J/kg — specific enthalpy of steam leaving


@dataclass
class EvaporatorBankState:
    """Operating state of the convective evaporator bank."""

    flue_gas_temp_in: float  # K
    flue_gas_temp_out: float  # K
    heat_transferred: float  # W — heat raising steam from drum water


@dataclass
class EconomizerState:
    """
    Operating state of the economizer at a given time step.

    Tracks feedwater conditions and residual flue gas heat recovery.
    """

    water_temp_in: float  # K   — feedwater temperature entering economizer
    water_temp_out: float  # K  — feedwater temperature leaving (-> drum)
    flue_gas_temp_in: float  # K  — flue gas temperature entering economizer
    flue_gas_temp_out: float  # K — flue gas temperature leaving (stack)
    heat_transferred: float  # W  — heat actually transferred (energy-balanced)
    water_enthalpy_gain: float  # J/kg — enthalpy gain per kg of feedwater


class SuperheaterModel:
    """
    Superheater heat exchanger model.

    Takes saturated steam from the boiler drum and superheats it
    using the flue gas leaving the furnace — the hottest exchanger in the gas path.

    Key output: steam_enthalpy_out — with spray water it sets the turbine inlet state.
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
        cp_flue_gas: float = CP_FLUE_GAS,
    ) -> SuperheaterState:
        """
        Calculate superheater performance at given operating conditions.

        Args:
            pressure_pa: Steam drum pressure [Pa].
            steam_flow: Steam mass flow through superheater [kg/s].
            flue_gas_temp_in: Flue gas temperature entering superheater [K].
            flue_gas_flow: Flue gas mass flow [kg/s].
            cp_flue_gas: Specific heat of flue gas [J/(kg·K)].

        Returns:
            SuperheaterState with all calculated temperatures and heat transfer.
        """
        t_sat = properties.saturation_temperature(pressure_pa)
        h_steam_in = properties.vapor_enthalpy_at_pressure(pressure_pa)

        if steam_flow <= 0.0 or flue_gas_flow <= 0.0 or flue_gas_temp_in <= t_sat:
            return SuperheaterState(
                steam_temp_in=t_sat,
                steam_temp_out=t_sat,
                flue_gas_temp_in=flue_gas_temp_in,
                flue_gas_temp_out=flue_gas_temp_in,
                heat_transferred=0.0,
                steam_enthalpy_in=h_steam_in,
                steam_enthalpy_out=h_steam_in,
            )

        c_steam = steam_flow * properties.superheat_cp(pressure_pa)
        c_gas = flue_gas_flow * cp_flue_gas
        effectiveness = counterflow_effectiveness(self.ua, c_steam, c_gas)
        q_actual = effectiveness * min(c_steam, c_gas) * (flue_gas_temp_in - t_sat)

        return SuperheaterState(
            steam_temp_in=t_sat,
            steam_temp_out=t_sat + q_actual / c_steam,
            flue_gas_temp_in=flue_gas_temp_in,
            flue_gas_temp_out=flue_gas_temp_in - q_actual / c_gas,
            heat_transferred=q_actual,
            steam_enthalpy_in=h_steam_in,
            steam_enthalpy_out=h_steam_in + q_actual / steam_flow,
        )


class EvaporatorBankModel:
    """
    Convective evaporator bank between the superheater and the economizer.

    The water side boils at saturation temperature, so its capacity rate is unbounded
    and the effectiveness reduces to 1 − exp(−NTU).
    """

    def __init__(
        self,
        area: float = BANK_HEAT_TRANSFER_AREA,
        htc: float = BANK_OVERALL_HTC,
    ) -> None:
        self.ua = area * htc  # W/K

    def calculate(
        self,
        saturation_temp: float,
        flue_gas_temp_in: float,
        flue_gas_flow: float,
        cp_flue_gas: float = CP_FLUE_GAS,
    ) -> EvaporatorBankState:
        """Heat raised into the drum circuit and the gas temperature leaving."""
        c_gas = flue_gas_flow * cp_flue_gas
        if c_gas <= 0.0 or flue_gas_temp_in <= saturation_temp:
            return EvaporatorBankState(flue_gas_temp_in, flue_gas_temp_in, 0.0)
        effectiveness = 1.0 - math.exp(-self.ua / c_gas)
        q_actual = effectiveness * c_gas * (flue_gas_temp_in - saturation_temp)
        return EvaporatorBankState(
            flue_gas_temp_in=flue_gas_temp_in,
            flue_gas_temp_out=flue_gas_temp_in - q_actual / c_gas,
            heat_transferred=q_actual,
        )


class EconomizerModel:
    """
    Economizer (feedwater preheater) heat exchanger model.

    Recovers residual heat from flue gas leaving the evaporator bank
    and uses it to preheat feedwater before it enters the drum.

    Energy balance is strictly enforced: if the subcooling limit clamps
    the water outlet temperature, q_actual and t_gas_out are recalculated
    from the clamped delta — no energy disappears into a 'black hole'.
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
        cp_flue_gas: float = CP_FLUE_GAS,
    ) -> EconomizerState:
        """
        Calculate economizer performance at given operating conditions.

        Energy balance is always consistent: q_water_absorbed == q_gas_released,
        and `water_enthalpy_gain` is exactly q_actual per kg of feedwater.

        Args:
            feedwater_flow: Feedwater mass flow [kg/s].
            feedwater_temp_in: Feedwater inlet temperature [K].
            pressure_pa: Drum pressure [Pa] — used to find saturation temp.
            flue_gas_temp_in: Flue gas temperature entering economizer [K].
            flue_gas_flow: Flue gas mass flow [kg/s].
            cp_flue_gas: Specific heat of flue gas [J/(kg·K)].

        Returns:
            EconomizerState with energy-balanced temperatures and heat transfer.
        """
        if (
            feedwater_flow <= 0.0
            or flue_gas_flow <= 0.0
            or flue_gas_temp_in <= feedwater_temp_in
        ):
            return EconomizerState(
                water_temp_in=feedwater_temp_in,
                water_temp_out=feedwater_temp_in,
                flue_gas_temp_in=flue_gas_temp_in,
                flue_gas_temp_out=flue_gas_temp_in,
                heat_transferred=0.0,
                water_enthalpy_gain=0.0,
            )

        # Feedwater must not reach saturation inside the economizer (steaming).
        t_sat = properties.saturation_temperature(pressure_pa)
        t_water_max = max(t_sat - ECO_SUBCOOLING_MARGIN, feedwater_temp_in)

        cp_water = properties.liquid_cp(0.5 * (feedwater_temp_in + t_water_max))
        c_water = feedwater_flow * cp_water
        c_gas = flue_gas_flow * cp_flue_gas

        effectiveness = counterflow_effectiveness(self.ua, c_water, c_gas)
        q_actual = (
            effectiveness * min(c_water, c_gas) * (flue_gas_temp_in - feedwater_temp_in)
        )
        t_water_out = feedwater_temp_in + q_actual / c_water

        if t_water_out > t_water_max:
            t_water_out = t_water_max
            q_actual = c_water * (t_water_out - feedwater_temp_in)

        return EconomizerState(
            water_temp_in=feedwater_temp_in,
            water_temp_out=t_water_out,
            flue_gas_temp_in=flue_gas_temp_in,
            flue_gas_temp_out=flue_gas_temp_in - q_actual / c_gas,
            heat_transferred=q_actual,
            water_enthalpy_gain=q_actual / feedwater_flow,
        )
