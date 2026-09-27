"""JSON payloads from the broker, decoded so that no malformed one can escape as a raise.

Four services had their own "UTF-8, then json.loads, then it must be an object", each
catching a different subset of what can go wrong. An exception that got past a handler
reached the MQTT session and was taken for a broker outage: `NaN` (which `json.loads`
accepts and a strict encoder then refuses), nesting deep enough for a `RecursionError`,
an integer too long to parse. Here every one of those is simply "not a payload".
"""

from __future__ import annotations

import json
import math
from typing import Any, NoReturn

MAX_JSON_PAYLOAD_BYTES = 64 * 1024


def _refuse_constant(literal: str) -> NoReturn:
    raise ValueError(f"non-finite number {literal} in JSON")


def _finite_float(literal: str) -> float:
    number = float(literal)
    if not math.isfinite(number):
        raise ValueError(f"number {literal} overflows a float")
    return number


def decode_json_object(
    payload: object, *, max_bytes: int = MAX_JSON_PAYLOAD_BYTES
) -> dict[str, Any] | None:
    """The JSON object in `payload`, or None for anything that is not one.

    `payload` is bytes or text; for text the limit counts characters. A payload over
    `max_bytes` is refused before it is parsed, so its size costs nothing on the loop.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")
    if not isinstance(payload, bytes | bytearray | str):
        return None
    if len(payload) > max_bytes:
        return None
    try:
        value = json.loads(
            payload, parse_constant=_refuse_constant, parse_float=_finite_float
        )
    except ValueError, RecursionError, OverflowError:
        # ValueError covers JSONDecodeError, UnicodeDecodeError and the digit limit on
        # integers; RecursionError is deep nesting.
        return None
    return value if isinstance(value, dict) else None


def finite_number(value: object) -> float | None:
    """`value` as a float when it is a finite JSON number, else None.

    A bool is an int in Python but never a number in a payload; an int too large for a
    float, NaN and the infinities are refused rather than stored.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
