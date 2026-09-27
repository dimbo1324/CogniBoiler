"""Numeric guards shared by the control and protection code.

Every comparison with NaN is false, so `max(low, min(high, nan))` quietly returns `high`:
a reading that is not a number would become a fully open valve. These helpers make that
case explicit instead.
"""

from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    """`value` limited to [low, high]; NaN is refused, never mapped onto a bound."""
    if math.isnan(value):
        raise ValueError(f"cannot clamp NaN to [{low}, {high}]")
    return max(low, min(high, value))


def all_finite(*values: float) -> bool:
    """True when every value is a real number: neither NaN nor infinite."""
    return all(math.isfinite(value) for value in values)
