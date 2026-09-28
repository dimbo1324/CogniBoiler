"""Flux string literals, for every query and task a service builds from names.

The gateway builds history and KPI queries from bucket, measurement and field names; the
historian installs a downsampling task built from bucket and organisation names. Both
had their own complete copy of this escaper; this is that function once.
"""

from __future__ import annotations

_FLUX_ESCAPES: dict[str, str] = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def flux_string(value: str) -> str:
    """A Flux string literal, quoted and escaped.

    A quote in a name would otherwise end the literal, and Flux reads `${...}` inside a
    string as an expression. Flux knows only the escapes above; any other control
    character is refused with a ValueError rather than passed on.
    """
    parts: list[str] = []
    for index, char in enumerate(value):
        if char in _FLUX_ESCAPES:
            parts.append(_FLUX_ESCAPES[char])
        elif char == "$" and value[index + 1 : index + 2] == "{":
            parts.append("\\$")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            raise ValueError(f"control character {char!r} in a Flux string")
        else:
            parts.append(char)
    return '"' + "".join(parts) + '"'
