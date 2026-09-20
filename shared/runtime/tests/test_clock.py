"""The moment, in the unit every contract of this platform uses."""

from __future__ import annotations

import time

from cogniboiler_runtime.clock import (
    MILLISECONDS_PER_DAY,
    MILLISECONDS_PER_SECOND,
    NANOSECONDS_PER_MILLISECOND,
    now_ms,
)


def test_it_answers_in_milliseconds_of_the_current_moment() -> None:
    before = time.time()
    taken = now_ms()
    after = time.time()
    assert isinstance(taken, int)
    assert (
        before * MILLISECONDS_PER_SECOND - 1
        <= taken
        <= after * MILLISECONDS_PER_SECOND + 1
    )


def test_it_keeps_the_seconds_rather_than_rounding_them_away() -> None:
    # Two moments a hundredth of a second apart must not read as the same moment: on a
    # project where several things happen in one second, that is the whole question.
    first = now_ms()
    time.sleep(0.01)
    assert now_ms() > first


def test_the_units_are_the_ones_the_contracts_use() -> None:
    assert MILLISECONDS_PER_SECOND == 1_000
    assert MILLISECONDS_PER_DAY == 24 * 60 * 60 * MILLISECONDS_PER_SECOND
    assert NANOSECONDS_PER_MILLISECOND == 1_000_000
