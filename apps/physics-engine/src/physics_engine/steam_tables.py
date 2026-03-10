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

from iapws import IAPWS97


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
    temp_k = max(273.16, min(temp_k, 647.0))
    state = IAPWS97(T=temp_k, x=0.0)  # x=0: saturated liquid
    return _pa(state.P)


def saturation_temp(pressure_pa: float) -> float:
    """
    Saturation temperature at given pressure [K].

    Valid range: 611.7 Pa to 22.064 MPa (critical pressure).
    """
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(state.T)


# ─── Liquid water properties ──────────────────────────────────────────────────


def water_density(temp_k: float, pressure_pa: float) -> float:
    """
    Density of liquid water [kg/m³] at given T and P.

    Falls back to saturated liquid if state is two-phase.
    """
    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        if state.phase in ("Liquid", "Subcooled liquid", "Compressed liquid"):
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
    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        cp = state.cp
        if cp is not None and cp > 0:
            return float(cp) * 1000.0  # kJ/(kg·K) → J/(kg·K)
    except Exception:
        pass
    return 4186.0  # fallback: standard value at ~20°C


def water_enthalpy(temp_k: float, pressure_pa: float) -> float:
    """
    Specific enthalpy of liquid water [J/kg] at given T and P.
    """
    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.h) * 1000.0  # kJ/kg → J/kg
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
    try:
        state = IAPWS97(T=temp_k, P=_mpa(pressure_pa))
        return float(state.h) * 1000.0  # kJ/kg → J/kg
    except Exception:
        pass
    # fallback: saturated steam enthalpy
    sat = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(sat.h) * 1000.0


def steam_density(temp_k: float, pressure_pa: float) -> float:
    """
    Density of superheated steam [kg/m³] at given T and P.
    """
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
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    liquid = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    vapor = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return (float(vapor.h) - float(liquid.h)) * 1000.0  # kJ/kg → J/kg


# ─── Convenience: saturation line properties ──────────────────────────────────


def saturated_liquid_enthalpy(pressure_pa: float) -> float:
    """Enthalpy of saturated liquid [J/kg] at given pressure."""
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=0.0)
    return float(state.h) * 1000.0


def saturated_vapor_enthalpy(pressure_pa: float) -> float:
    """Enthalpy of saturated vapor [J/kg] at given pressure."""
    pressure_pa = max(611.7, min(pressure_pa, 22.064e6))
    state = IAPWS97(P=_mpa(pressure_pa), x=1.0)
    return float(state.h) * 1000.0
