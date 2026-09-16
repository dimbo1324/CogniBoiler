"""
Engineering units as OPC UA EUInformation (IEC 62541-8), coded with UNECE Recommendation
20 common codes. A unit without a code still carries its display name, with UnitId -1 as
the specification allows.
"""

from __future__ import annotations

from asyncua import ua

UNECE_NAMESPACE = "http://www.opcfoundation.org/UA/units/un/cefact"

_UNECE_CODES: dict[str, tuple[str, str]] = {
    "Pa": ("PAL", "pascal"),
    "m": ("MTR", "metre"),
    "K": ("KEL", "kelvin"),
    "J": ("JOU", "joule"),
    "W": ("WTT", "watt"),
    "kg/s": ("KGS", "kilogram per second"),
    "J/kg": ("J2", "joule per kilogram"),
    "s": ("SEC", "second"),
    "h": ("HUR", "hour"),
    "%": ("P1", "percent"),
    "1": ("C62", "one"),
    "ppmv": ("59", "part per million"),
}

_DESCRIPTIONS: dict[str, str] = {
    "J/J": "joule per joule",
    "kg/J": "kilogram per joule",
    "kg/MWh": "kilogram per megawatt hour",
}


def unit_id(code: str) -> int:
    """UnitId of a UNECE common code: its characters as a big-endian integer."""
    value = 0
    for char in code:
        value = (value << 8) | ord(char)
    return value


def engineering_units(unit: str) -> ua.EUInformation | None:
    """EUInformation for a unit; None for text and flags ("-")."""
    if unit == "-":
        return None
    info = ua.EUInformation()
    info.NamespaceUri = UNECE_NAMESPACE
    code = _UNECE_CODES.get(unit)
    if code is not None:
        info.UnitId = unit_id(code[0])
        info.Description = ua.LocalizedText(code[1])
    else:
        info.UnitId = -1
        info.Description = ua.LocalizedText(_DESCRIPTIONS.get(unit, unit))
    info.DisplayName = ua.LocalizedText(unit)
    return info
