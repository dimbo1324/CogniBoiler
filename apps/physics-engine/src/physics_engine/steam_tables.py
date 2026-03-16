"""
Steam property lookup using IAPWS-IF97 industrial standard.

All inputs and outputs use SI units:
    Temperature : Kelvin  (K)
    Pressure    : Pascal  (Pa)
    Density     : kg/m³
    Enthalpy    : J/kg
    Specific heat: J/(kg·K)

IAPWS-IF97 covers:
    Region 1: compressed liquid water
    Region 2: superheated steam
    Region 3: near-critical region
    Region 4: saturation line (two-phase)
    Region 5: high-temperature steam
"""

# ─── Monkey-patch: fix iapws _Region3 for numpy 2.x / Python 3.14 ────────────
#
# Root cause: inside iapws _Region3, scipy.optimize.fsolve passes the density
# argument `rho` as a 1-D numpy array (e.g. array([569.6...])), but the
# function then calls math.log(d) where d = rho / rhoc.  In numpy < 2.0
# math.log silently accepted 0-d and 1-element 1-D arrays; in numpy 2.x it
# raises "only 0-dimensional arrays can be converted to Python scalars".
#
# Fix: wrap _Region3 to ensure rho and T are always plain Python floats before
# delegating to the original implementation.  This is safe because _Region3 is
# a pure function that only takes scalar arguments.
# ─────────────────────────────────────────────────────────────────────────────
from typing import Any, cast

import iapws.iapws97 as _iapws97  # noqa: E402
import numpy as np
from iapws import IAPWS97

# iapws has no stubs — attribute resolves to Any, no annotation needed
_orig_region3 = _iapws97._Region3  # noqa: SLF001


# N802 fix: function name lowercase
# N803 fix: argument name lowercase (t instead of T)
def _patched_region3(rho: object, t: object) -> dict[str, Any]:
    """Scalar-safe wrapper around iapws._Region3."""
    # N806 fix: local variable lowercase (t_f instead of T)
    rho_f: float
    t_f: float
    # isinstance narrows rho to np.ndarray -> .flat[0] is safe
    if isinstance(rho, np.ndarray):
        rho_f = float(rho.flat[0])
    else:
        # cast(Any, ...) lets float() accept an arbitrary object
        rho_f = float(cast(Any, rho))
    if isinstance(t, np.ndarray):
        t_f = float(t.flat[0])
    else:
        t_f = float(cast(Any, t))
    # cast fixes "Returning Any from function declared to return dict[str, Any]"
    return cast(dict[str, Any], _orig_region3(rho_f, t_f))


# no type: ignore needed — iapws module resolves to Any, assignment is accepted
_iapws97._Region3 = _patched_region3  # noqa: SLF001

# ─────────────────────────────────────────────────────────────────────────────


def _to_float(value: object) -> float:
    """Ensure value is a plain Python float before passing to iapws."""
    if isinstance(value, np.generic | np.ndarray):
        return float(value)
    return float(cast(Any, value))


def _mpa(pressure_pa: float) -> float:
    """Convert Pa to MPa for iapws API."""
    return pressure_pa / 1.0e6


def _pa(pressure_mpa: float) -> float:
    """Convert MPa to Pa."""
    return pressure_mpa * 1.0e6


# ─── Saturation properties ────────────────────────────────────────────────────


def saturation_pressure(temp_k: float) -> float:
    """
    Saturation pressure at given temperature [Pa].

    Valid range: 273.15 K to 647.096 K (critical point).
    """
    temp_k = _to_float(temp_k)
    temp_k = max(273.16, min(temp_k, 647.0))
    state = IAPWS97(T=temp_k, x=0.0)  # x=0: saturated liquid
    return _pa(state.P)


def saturation_temp(pressure_pa: float) -> float:
    """
    Saturation temperature at given pressure [K].

    Valid range: 611.7 Pa to 22.064 MPa (critical pressure).
    """
    pressure_pa = _to_float(pressure_pa)
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(state.T)


# ─── Liquid water properties ──────────────────────────────────────────────────


def water_density(temp_k: float, pressure_pa: float) -> float:
    """
    Density of liquid water [kg/m³] at given T and P.

    Falls back to saturated liquid if state is two-phase.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)
    # Clamp to liquid region — avoid supercritical (Region 3)
    temp_k = min(temp_k, 623.0)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        if state.phase in ("liq", "Liquid", "Subcooled liquid", "Compressed liquid"):
            return float(state.rho)
    except Exception:
        pass
    # fallback: saturated liquid density
    sat = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(sat.rho)


def water_specific_heat(temp_k: float, pressure_pa: float) -> float:
    """
    Specific heat capacity of liquid water Cp [J/(kg·K)] at given T and P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)
    # Clamp to liquid region — avoid supercritical (Region 3)
    temp_k = min(temp_k, 623.0)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        cp = state.cp
        if cp is not None and cp > 0:
            return float(cp) * 1000.0  # kJ/(kg·K) -> J/(kg·K)
    except Exception:
        pass
    return 4186.0  # fallback: standard value at ~20°C


def water_enthalpy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific enthalpy of liquid water [J/kg] at given T and P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)
    # Clamp to liquid region — avoid supercritical (Region 3)
    temp_k = min(temp_k, 623.0)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.h) * 1000.0  # kJ/kg -> J/kg
    except Exception:
        pass
    sat = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(sat.h) * 1000.0


# ─── Steam properties ─────────────────────────────────────────────────────────


def steam_enthalpy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific enthalpy of superheated steam [J/kg] at given T and P.

    For superheated steam: T must be above saturation temperature at P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.h) * 1000.0  # kJ/kg -> J/kg
    except Exception:
        pass
    # fallback: saturated steam enthalpy
    sat = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(sat.h) * 1000.0


def steam_density(temp_k: float, pressure_pa: float) -> float:
    """
    Density of superheated steam [kg/m³] at given T and P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.rho)
    except Exception:
        pass
    sat = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(sat.rho)


def steam_specific_heat(temp_k: float, pressure_pa: float) -> float:
    """
    Specific heat capacity of steam Cp [J/(kg·K)] at given T and P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        cp = state.cp
        if cp is not None and cp > 0:
            return float(cp) * 1000.0
    except Exception:
        pass
    return 2010.0  # fallback: standard superheated steam value


def latent_heat(pressure_pa: float) -> float:
    """
    Latent heat of vaporization [J/kg] at given pressure.

    Difference between saturated vapor and saturated liquid enthalpy.
    """
    pressure_pa = _to_float(pressure_pa)
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    liquid = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    vapor = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return (float(vapor.h) - float(liquid.h)) * 1000.0  # kJ/kg -> J/kg


# ─── Convenience: saturation line properties ──────────────────────────────────


def saturated_liquid_enthalpy(pressure_pa: float) -> float:
    """Enthalpy of saturated liquid [J/kg] at given pressure."""
    pressure_pa = _to_float(pressure_pa)
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(state.h) * 1000.0


def saturated_vapor_enthalpy(pressure_pa: float) -> float:
    """Enthalpy of saturated vapor [J/kg] at given pressure."""
    pressure_pa = _to_float(pressure_pa)
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(state.h) * 1000.0


# ───────────────────────────────────────────────────────────────


def steam_entropy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific entropy of superheated steam [J/(kg·K)] at given T and P.

    Used as the inlet condition for isentropic turbine expansion:
        s_in = steam_entropy(T_in, P_in)

    Valid range: T above saturation at P (superheated region).
    """

    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)

    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.s) * 1000.0  # kJ/(kg·K) -> J/(kg·K)
    except Exception:
        pass
    # fallback: saturated vapor entropy
    sat = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(sat.s) * 1000.0


def isentropic_enthalpy(entropy_in: float, pressure_out: float) -> float:
    """
    Specific enthalpy of steam after isentropic expansion [J/kg].

    Finds the state at (s=entropy_in, P=pressure_out) — i.e. the outlet
    condition of an ideal turbine stage.  Used to compute isentropic work:
        W_ideal = h_in − isentropic_enthalpy(s_in, P_out)

    Args:
        entropy_in:   Inlet entropy [J/(kg·K)].
        pressure_out: Turbine exhaust pressure [Pa].

    Returns:
        Specific enthalpy at isentropic outlet [J/kg].
    """
    entropy_in = _to_float(entropy_in)
    pressure_out = _to_float(pressure_out)

    s_mpa = entropy_in / 1000.0  # J/(kg·K) -> kJ/(kg·K) for iapws

    try:
        state = IAPWS97(P=_mpa(pressure_out), s=s_mpa)
        return float(state.h) * 1000.0  # kJ/kg -> J/kg
    except Exception:
        pass
    # fallback: saturated vapor enthalpy at exhaust pressure
    sat = IAPWS97(P=_mpa(pressure_out), x=1.0)
    return float(sat.h) * 1000.0


def exhaust_temp(enthalpy: float, pressure_pa: float) -> float:
    """
    Temperature of steam at given enthalpy and pressure [K].

    Used to find turbine exhaust temperature from actual outlet enthalpy.
    Works for both wet and superheated exhaust conditions.

    Args:
        enthalpy:    Specific enthalpy [J/kg].
        pressure_pa: Exhaust pressure [Pa].

    Returns:
        Temperature [K].
    """
    enthalpy = _to_float(enthalpy)
    pressure_pa = _to_float(pressure_pa)

    try:
        state = IAPWS97(P=_mpa(pressure_pa), h=enthalpy / 1000.0)  # J/kg -> kJ/kg
        return float(state.T)
    except Exception:
        pass
    # fallback: saturation temperature at exhaust pressure
    sat = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(sat.T)
