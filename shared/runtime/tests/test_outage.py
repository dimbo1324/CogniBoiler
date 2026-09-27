"""One warning when a dependency goes away, one line when it is back, nothing between."""

from __future__ import annotations

import logging

import pytest
from cogniboiler_runtime.outage import OutageLog

LOGGER = "test.outage"


def levels(caplog: pytest.LogCaptureFixture) -> list[int]:
    return [record.levelno for record in caplog.records if record.name == LOGGER]


class TestOutageLog:
    def test_an_outage_costs_one_warning_however_long_it_lasts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        outage = OutageLog(logging.getLogger(LOGGER), "InfluxDB")
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            first = outage.failed(ConnectionError("refused"))
            repeats = [outage.failed(ConnectionError("refused")) for _ in range(5)]

        assert first is True
        assert repeats == [False] * 5
        assert levels(caplog) == [logging.WARNING] + [logging.DEBUG] * 5
        warning = caplog.records[0].getMessage()
        assert "InfluxDB" in warning
        assert "refused" in warning
        assert outage.down is True

    def test_the_recovery_is_said_once(self, caplog: pytest.LogCaptureFixture) -> None:
        outage = OutageLog(logging.getLogger(LOGGER), "PLCService")
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            outage.failed(TimeoutError("deadline"))
            back = outage.recovered()
            again = outage.recovered()

        assert (back, again) == (True, False)
        assert levels(caplog) == [logging.WARNING, logging.INFO]
        assert "PLCService" in caplog.records[1].getMessage()
        assert outage.down is False

    def test_a_dependency_that_never_failed_says_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        outage = OutageLog(logging.getLogger(LOGGER), "AlarmService")
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert outage.recovered() is False
        assert levels(caplog) == []
        assert outage.down is False

    def test_a_second_outage_warns_again(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        outage = OutageLog(logging.getLogger(LOGGER), "telemetry")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            outage.failed(OSError("one"))
            outage.recovered()
            outage.failed(OSError("two"))
        assert levels(caplog) == [logging.WARNING, logging.WARNING]
