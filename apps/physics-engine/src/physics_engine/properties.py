"""
Fast water and steam properties for the live plant model.

`steam_tables` evaluates IAPWS-IF97 through the iapws package, which costs about half a
millisecond per call; one integration step of the plant needs dozens of calls, so the live
runtime could not run faster than real time. The plant only needs properties along the
saturation line (drum water, condenser, feedwater) and a secant heat capacity of
superheated steam, so these are tabulated once from IAPWS-IF97 and linearly interpolated.
`steam_tables` stays the exact reference for turbine expansion and offline checks.

Subcooled feedwater is represented by saturated liquid at the same temperature; at drum
pressure that understates enthalpy by about 1.5 %, well inside the model's accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from iapws import IAPWS97

T_TABLE_MIN_K: float = 275.0
T_TABLE_MAX_K: float = 645.0
T_TABLE_STEP_K: float = 0.5

# Superheated-steam secant heat capacity is taken between saturation and this design
# outlet temperature, which is what the superheater effectiveness-NTU model needs.
SUPERHEAT_REFERENCE_TEMP_K: float = 811.0


@dataclass(frozen=True)
class _SaturationTable:
    temp_k: np.ndarray
    log_pressure: np.ndarray
    liquid_density: np.ndarray
    liquid_cp: np.ndarray
    liquid_enthalpy: np.ndarray
    vapor_enthalpy: np.ndarray
    superheat_cp: np.ndarray


@cache
def _table() -> _SaturationTable:
    temps = np.arange(T_TABLE_MIN_K, T_TABLE_MAX_K + T_TABLE_STEP_K / 2, T_TABLE_STEP_K)
    size = temps.size
    log_p = np.empty(size)
    rho_f = np.empty(size)
    cp_f = np.empty(size)
    h_f = np.empty(size)
    h_g = np.empty(size)
    cp_sh = np.empty(size)
    for i, temp in enumerate(temps):
        liquid = IAPWS97(T=float(temp), x=0.0)
        vapor = IAPWS97(T=float(temp), x=1.0)
        pressure_mpa = float(liquid.P)
        log_p[i] = math.log(pressure_mpa * 1.0e6)
        rho_f[i] = float(liquid.rho)
        cp_f[i] = float(liquid.cp) * 1000.0
        h_f[i] = float(liquid.h) * 1000.0
        h_g[i] = float(vapor.h) * 1000.0
        superheated = IAPWS97(T=SUPERHEAT_REFERENCE_TEMP_K, P=pressure_mpa)
        cp_sh[i] = (float(superheated.h) * 1000.0 - h_g[i]) / (
            SUPERHEAT_REFERENCE_TEMP_K - float(temp)
        )
    return _SaturationTable(
        temp_k=temps,
        log_pressure=log_p,
        liquid_density=rho_f,
        liquid_cp=cp_f,
        liquid_enthalpy=h_f,
        vapor_enthalpy=h_g,
        superheat_cp=cp_sh,
    )


def warm_up() -> None:
    """Build the tables now instead of on the first integration step."""
    _table()


def _clamp_temp(temp_k: float) -> float:
    return min(max(temp_k, T_TABLE_MIN_K), T_TABLE_MAX_K)


def _at_temp(values: np.ndarray, temp_k: float) -> float:
    table = _table()
    return float(np.interp(_clamp_temp(temp_k), table.temp_k, values))


def saturation_pressure(temp_k: float) -> float:
    """Saturation pressure [Pa] at a temperature [K]."""
    return math.exp(_at_temp(_table().log_pressure, temp_k))


def saturation_temperature(pressure_pa: float) -> float:
    """Saturation temperature [K] at a pressure [Pa]."""
    table = _table()
    log_p = math.log(max(pressure_pa, 1.0))
    return float(np.interp(log_p, table.log_pressure, table.temp_k))


def liquid_density(temp_k: float) -> float:
    """Density of saturated liquid water [kg/m³] at a temperature [K]."""
    return _at_temp(_table().liquid_density, temp_k)


def liquid_cp(temp_k: float) -> float:
    """Isobaric heat capacity of saturated liquid water [J/(kg·K)]."""
    return _at_temp(_table().liquid_cp, temp_k)


def liquid_enthalpy(temp_k: float) -> float:
    """Specific enthalpy of saturated liquid water [J/kg] at a temperature [K]."""
    return _at_temp(_table().liquid_enthalpy, temp_k)


def vapor_enthalpy_at_pressure(pressure_pa: float) -> float:
    """Specific enthalpy of saturated steam [J/kg] at a pressure [Pa]."""
    return _at_temp(_table().vapor_enthalpy, saturation_temperature(pressure_pa))


def liquid_enthalpy_at_pressure(pressure_pa: float) -> float:
    """Specific enthalpy of saturated liquid [J/kg] at a pressure [Pa]."""
    return _at_temp(_table().liquid_enthalpy, saturation_temperature(pressure_pa))


def superheat_cp(pressure_pa: float) -> float:
    """Secant heat capacity of steam [J/(kg·K)] from saturation to the design outlet."""
    return _at_temp(_table().superheat_cp, saturation_temperature(pressure_pa))
