"""The Flux query behind GET /api/v1/history, and what may reach it as a name."""

from __future__ import annotations

import re

import pytest
from api_gateway.clients import build_history_query, build_kpi_query, flux_string


def test_range_bounds_are_integer_nanoseconds() -> None:
    flux = build_history_query(
        bucket="sensors",
        measurement="boiler_sensors",
        start_ms=1_789_000_000_123,
        end_ms=1_789_000_900_456,
        limit=5,
    )
    assert "start: time(v: 1789000000123000000)" in flux
    assert "stop: time(v: 1789000900456000000)" in flux
    assert not re.search(r"time\(v: -?\d+\.\d", flux)


def test_bucket_measurement_and_limit_reach_the_query() -> None:
    flux = build_history_query(
        bucket="sensors",
        measurement="turbine_sensors",
        start_ms=0,
        end_ms=1,
        limit=200,
    )
    assert 'from(bucket: "sensors")' in flux
    assert 'r._measurement == "turbine_sensors"' in flux
    assert "limit(n: 200)" in flux


class TestNamesCannotLeaveTheirLiteral:
    """The route validates its parameters; the query builder does not rely on that.

    A name carrying a quote, a backslash or a newline would otherwise end the string
    literal it is written into, and whatever followed would be read as Flux.
    """

    @pytest.mark.parametrize(
        "hostile",
        [
            'boiler_sensors" or true or "',
            'boiler_sensors" |> yield(name: "leak',
            "boiler_sensors" + chr(92),
            "boiler_sensors" + chr(10) + 'from(bucket: "secrets")',
        ],
    )
    def test_a_hostile_measurement_stays_one_string(self, hostile: str) -> None:
        flux = build_history_query(
            bucket="sensors",
            measurement=hostile,
            start_ms=0,
            end_ms=1,
            limit=10,
        )
        # The payload is in the query as data: one escaped literal, whatever it carried.
        assert flux_string(hostile) in flux
        # With that one literal taken out, the query is still the query it was meant to
        # be: one source, one filter, and no clause the payload smuggled in.
        skeleton = flux.replace(flux_string(hostile), '"measurement"')
        assert skeleton.count("|> filter(") == 1
        assert skeleton.count("from(bucket:") == 1

    def test_a_hostile_field_name_stays_one_string(self) -> None:
        hostile = 'pressure_pa" or r._field == "secret'
        flux = build_history_query(
            bucket="sensors",
            measurement="boiler_sensors",
            start_ms=0,
            end_ms=1,
            limit=10,
            fields=(hostile,),
        )
        # The measurement filter and the field filter, and nothing else.
        assert flux.count("|> filter(") == 2
        assert flux_string(hostile) in flux
        assert hostile not in flux

    def test_a_bucket_name_from_the_environment_is_escaped_too(self) -> None:
        hostile = 'sensors" |> yield(name: "leak'
        flux = build_history_query(
            bucket=hostile,
            measurement="boiler_sensors",
            start_ms=0,
            end_ms=1,
            limit=10,
        )
        assert flux_string(hostile) in flux
        assert hostile not in flux
        kpi = build_kpi_query(bucket='sensors"', start_ms=0, end_ms=1, aggregated=False)
        # Four sources, each with its bucket still inside its own literal.
        assert kpi.count("from(bucket:") == 4
        assert kpi.count(flux_string('sensors"')) == 4

    def test_the_escape_is_the_flux_spelling_of_a_string(self) -> None:
        quote, backslash = '"', chr(92)
        assert flux_string("sensors") == quote + "sensors" + quote
        assert (
            flux_string("a" + quote + "b")
            == quote + "a" + backslash + quote + "b" + quote
        )
        assert flux_string("line" + chr(10)) == quote + "line" + backslash + "n" + quote
        # Non-ASCII stays readable rather than turning into escapes.
        assert flux_string("датчик") == quote + "датчик" + quote
