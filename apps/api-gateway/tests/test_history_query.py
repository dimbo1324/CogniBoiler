"""The Flux query behind GET /api/v1/history."""

from __future__ import annotations

import re

from api_gateway.clients import build_history_query


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
