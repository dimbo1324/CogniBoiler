"""The Flux string literal both the gateway's queries and the historian's task use."""

from __future__ import annotations

import pytest
from cogniboiler_runtime.flux import flux_string


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        ("sensors", '"sensors"'),
        ('a"b', '"a\\"b"'),
        ("a\\b", '"a\\\\b"'),
        ("a${r._value}b", '"a\\${r._value}b"'),
        ("cost $5", '"cost $5"'),
        ("a$", '"a$"'),
        ("a\nb\rc\td", '"a\\nb\\rc\\td"'),
        # Non-ASCII stays readable rather than turning into escapes.
        ("Kraftwerk Süd", '"Kraftwerk Süd"'),
        ("датчик", '"датчик"'),
        ("", '""'),
    ],
)
def test_a_flux_literal_escapes_what_flux_would_read(value: str, literal: str) -> None:
    assert flux_string(value) == literal


@pytest.mark.parametrize("value", ["a\x01b", "a\x00", "a\x7f", "a\x1bb"])
def test_other_control_characters_are_refused(value: str) -> None:
    # Flux has no \uXXXX escape, which is what json.dumps writes for them.
    with pytest.raises(ValueError, match="control character"):
        flux_string(value)
