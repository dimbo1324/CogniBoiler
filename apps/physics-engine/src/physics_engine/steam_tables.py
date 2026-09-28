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
# a pure function that only takes scalar arguments. It is still a global patch of a
# private iapws function applied at import: re-check it on every iapws upgrade.
# ─────────────────────────────────────────────────────────────────────────────
import logging
import math
from typing import Any, cast

import iapws.iapws97 as _iapws97
import numpy as np
from iapws import IAPWS97

from physics_engine.metrics import PROPERTY_FALLBACKS

_orig_region3 = _iapws97._Region3  # noqa: SLF001


def _patched_region3(rho: object, t: object) -> dict[str, Any]:
    """Scalar-safe wrapper around iapws._Region3."""
    rho_f: float
    t_f: float
    if isinstance(rho, np.ndarray):
        rho_f = float(rho.flat[0])
    else:
        rho_f = float(cast(Any, rho))
    if isinstance(t, np.ndarray):
        t_f = float(t.flat[0])
    else:
        t_f = float(cast(Any, t))
    return cast(dict[str, Any], _orig_region3(rho_f, t_f))


_iapws97._Region3 = _patched_region3  # noqa: SLF001

# ─────────────────────────────────────────────────────────────────────────────


# The IF97 saturation line runs from the triple point to the critical point. Liquid states
# are evaluated below LIQUID_T_MAX_K, away from the near-critical Region 3.
IF97_P_MIN_PA: float = 611.7
IF97_P_CRIT_PA: float = 22.064e6
IF97_T_MIN_K: float = 273.16
IF97_T_SAT_MAX_K: float = 647.0
LIQUID_T_MAX_K: float = 623.0

# What iapws raises for a state outside the formulation ("Incoming out of bound") or one
# its solvers cannot reach. Anything else is a defect in the call and propagates.
IAPWS_OUT_OF_RANGE: tuple[type[Exception], ...] = (
    NotImplementedError,
    ValueError,
    ArithmeticError,
)

_LIQUID_PHASES = ("liq", "Liquid", "Subcooled liquid", "Compressed liquid")

logger = logging.getLogger(__name__)
_warned: set[str] = set()


def _to_float(value: object) -> float:
    """A plain, finite Python float for iapws; NaN or infinity is refused."""
    if isinstance(value, np.generic | np.ndarray):
        # numpy 2 converts only 0-d arrays with float(); item() takes any size-1 array.
        number = float(value.item())
    else:
        number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError(f"non-finite property input {number!r}")
    return number


def _mpa(pressure_pa: float) -> float:
    """Convert Pa to MPa for iapws API."""
    return pressure_pa / 1.0e6


def _pa(pressure_mpa: float) -> float:
    """Convert MPa to Pa."""
    return pressure_mpa * 1.0e6


def _clamp_saturation_pressure(pressure_pa: float) -> float:
    return max(IF97_P_MIN_PA, min(pressure_pa, IF97_P_CRIT_PA))


def _saturated(pressure_pa: float, quality: float) -> Any:
    return IAPWS97(P=_mpa(_clamp_saturation_pressure(pressure_pa)), x=quality)


def _state_or_none(function: str, **inputs: float) -> Any:
    """The IF97 state at `inputs`, or None (logged and counted) when it is out of range.

    The caller then substitutes the saturated state at the same pressure. The first
    substitution per function is a warning; the rest are debug, but all are counted.
    """
    try:
        return IAPWS97(**inputs)
    except IAPWS_OUT_OF_RANGE as exc:
        PROPERTY_FALLBACKS.labels(function).inc()
        if function in _warned:
            logger.debug("%s fell back to saturation at %s: %s", function, inputs, exc)
        else:
            _warned.add(function)
            logger.warning(
                "%s fell back to saturation at %s (IF97 units: MPa, kJ): %s; "
                "further fallbacks are counted in physics_property_fallbacks_total",
                function,
                inputs,
                exc,
            )
        return None


# ─── Saturation properties ────────────────────────────────────────────────────


def saturation_pressure(temp_k: float) -> float:
    """
    Saturation pressure at given temperature [Pa].

    Valid range: 273.16 K to 647.0 K (just below the critical point); clamped to it.
    """
    temp_k = _to_float(temp_k)
    temp_k = max(IF97_T_MIN_K, min(temp_k, IF97_T_SAT_MAX_K))
    state = IAPWS97(T=temp_k, x=0.0)  # x=0: saturated liquid
    return _pa(state.P)


def saturation_temp(pressure_pa: float) -> float:
    """
    Saturation temperature at given pressure [K].

    Valid range: 611.7 Pa to 22.064 MPa (critical pressure); clamped to it.
    """
    pressure_pa = _clamp_saturation_pressure(_to_float(pressure_pa))
    state = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(state.T)


# ─── Liquid water properties ──────────────────────────────────────────────────


def water_density(temp_k: float, pressure_pa: float) -> float:
    """
    Density of liquid water [kg/m³] at given T and P.

    A state that is not liquid, or outside IF97, gives the saturated liquid density.
    """
    temp_k = min(_to_float(temp_k), LIQUID_T_MAX_K)
    pressure_pa = _to_float(pressure_pa)
    state = _state_or_none("water_density", T=temp_k, P=_mpa(pressure_pa))
    if state is None or state.phase not in _LIQUID_PHASES:
        state = _saturated(pressure_pa, 0.0)
    return float(state.rho)


def water_enthalpy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific enthalpy of liquid water [J/kg] at given T and P.
    """
    temp_k = min(_to_float(temp_k), LIQUID_T_MAX_K)
    pressure_pa = _to_float(pressure_pa)
    state = _state_or_none("water_enthalpy", T=temp_k, P=_mpa(pressure_pa))
    if state is None:
        state = _saturated(pressure_pa, 0.0)
    return float(state.h) * 1000.0  # kJ/kg -> J/kg


# ─── Steam properties ─────────────────────────────────────────────────────────


def steam_enthalpy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific enthalpy of superheated steam [J/kg] at given T and P.

    For superheated steam: T must be above saturation temperature at P.
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)
    state = _state_or_none("steam_enthalpy", T=temp_k, P=_mpa(pressure_pa))
    if state is None:
        state = _saturated(pressure_pa, 1.0)
    return float(state.h) * 1000.0  # kJ/kg -> J/kg


def steam_entropy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific entropy of superheated steam [J/(kg·K)] at given T and P.

    Used as the inlet condition for isentropic turbine expansion:
        s_in = steam_entropy(T_in, P_in)

    Valid range: T above saturation at P (superheated region).
    """
    temp_k = _to_float(temp_k)
    pressure_pa = _to_float(pressure_pa)
    state = _state_or_none("steam_entropy", T=temp_k, P=_mpa(pressure_pa))
    if state is None:
        state = _saturated(pressure_pa, 1.0)
    return float(state.s) * 1000.0  # kJ/(kg·K) -> J/(kg·K)


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
    state = _state_or_none(
        "isentropic_enthalpy", P=_mpa(pressure_out), s=entropy_in / 1000.0
    )
    if state is None:
        state = _saturated(pressure_out, 1.0)
    return float(state.h) * 1000.0  # kJ/kg -> J/kg


def steam_state_from_enthalpy(
    enthalpy: float, pressure_pa: float
) -> tuple[float, float]:
    """
    Temperature [K] and specific entropy [J/(kg·K)] of steam at (h, P).

    Used for the turbine inlet, where spray water mixed into superheated steam fixes
    the enthalpy rather than the temperature.
    """
    enthalpy = _to_float(enthalpy)
    pressure_pa = _to_float(pressure_pa)
    state = IAPWS97(P=_mpa(pressure_pa), h=enthalpy / 1000.0)
    return float(state.T), float(state.s) * 1000.0


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
    state = _state_or_none("exhaust_temp", P=_mpa(pressure_pa), h=enthalpy / 1000.0)
    if state is None:
        state = _saturated(pressure_pa, 1.0)
    return float(state.T)
