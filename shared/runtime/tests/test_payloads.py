"""Decoding a JSON payload from the broker: every malformed shape is a None, never a raise.

Each case below once escaped a service's handler and reached the MQTT session as if the
broker had failed (GW-API-11, HIST-01, ALM-04 in the audit).
"""

from __future__ import annotations

import math

import pytest
from cogniboiler_runtime.payloads import (
    MAX_JSON_PAYLOAD_BYTES,
    decode_json_object,
    finite_number,
)


class TestDecodeJsonObject:
    @pytest.mark.parametrize(
        "raw",
        [b'{"alarm_id": 7, "state": "active"}', bytearray(b'{"alarm_id": 7}')],
    )
    def test_an_object_is_decoded(self, raw: bytes | bytearray) -> None:
        decoded = decode_json_object(raw)
        assert decoded is not None
        assert decoded["alarm_id"] == 7

    def test_text_is_decoded_like_bytes(self) -> None:
        assert decode_json_object('{"kind": "mode_changed"}') == {
            "kind": "mode_changed"
        }

    @pytest.mark.parametrize(
        ("raw", "case"),
        [
            (b'{"value": NaN}', "NaN literal"),
            (b'{"value": Infinity}', "Infinity literal"),
            (b'{"value": -Infinity}', "-Infinity literal"),
            (b'{"value": 1e400}', "a float that overflows to inf"),
            (b'{"nested": {"value": NaN}}', "NaN below the top level"),
        ],
    )
    def test_a_non_finite_number_rejects_the_payload(
        self, raw: bytes, case: str
    ) -> None:
        assert decode_json_object(raw) is None, case

    @pytest.mark.parametrize(
        ("raw", "case"),
        [
            (b"[1, 2, 3]", "an array"),
            (b'"text"', "a string"),
            (b"42", "a number"),
            (b"null", "null"),
            (b"", "empty"),
            (b"{not json", "malformed"),
            (b"\xff\xfe{}", "not UTF-8"),
        ],
    )
    def test_anything_but_an_object_is_none(self, raw: bytes, case: str) -> None:
        assert decode_json_object(raw) is None, case

    def test_deep_nesting_is_none_instead_of_a_recursion_error(self) -> None:
        assert decode_json_object(b"[" * 100_000) is None
        assert decode_json_object(b'{"a":' * 100_000) is None

    def test_an_integer_past_the_digit_limit_is_none(self) -> None:
        assert decode_json_object(b'{"id": ' + b"1" * 5000 + b"}") is None

    def test_a_payload_over_the_limit_is_refused_before_it_is_parsed(self) -> None:
        padding = b" " * MAX_JSON_PAYLOAD_BYTES
        assert decode_json_object(b"{}" + padding) is None
        assert decode_json_object(b"{}" + b" " * 10, max_bytes=12) == {}
        assert decode_json_object(b"{}" + b" " * 11, max_bytes=12) is None

    def test_oversized_text_is_refused_too(self) -> None:
        assert decode_json_object("{}" + " " * MAX_JSON_PAYLOAD_BYTES) is None

    @pytest.mark.parametrize("raw", [None, 42, 1.5, object()])
    def test_a_payload_that_is_not_bytes_or_text_is_none(self, raw: object) -> None:
        assert decode_json_object(raw) is None

    def test_a_negative_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            decode_json_object(b"{}", max_bytes=-1)


class TestFiniteNumber:
    @pytest.mark.parametrize(
        ("value", "expected"), [(1, 1.0), (2.5, 2.5), (-0.0, 0.0), (10**20, 1e20)]
    )
    def test_a_finite_number_is_returned_as_a_float(
        self, value: object, expected: float
    ) -> None:
        number = finite_number(value)
        assert isinstance(number, float)
        assert number == expected

    @pytest.mark.parametrize(
        ("value", "case"),
        [
            (True, "a bool is an int in Python, not a number in a payload"),
            (False, "False"),
            (math.nan, "NaN"),
            (math.inf, "inf"),
            (-math.inf, "-inf"),
            (10**400, "an int too large for a float"),
            ("1.5", "a numeric string"),
            (None, "null"),
            ([1], "a list"),
        ],
    )
    def test_anything_else_is_none(self, value: object, case: str) -> None:
        assert finite_number(value) is None, case
