"""What the PLC says about itself: refusals in the log, publisher losses in metrics."""

from __future__ import annotations

import logging

import cogniboiler_pb2 as pb2
import pytest
from plc_controller.metrics import PlcCollector
from plc_controller.service import PLCService, RuntimeMode
from plc_fakes import RecordingPublisher, plc, state


def samples(svc: PLCService) -> dict[str, float]:
    return {
        sample.name: sample.value
        for family in PlcCollector(svc).collect()
        for sample in family.samples
        if not sample.labels
    }


def documentation(svc: PLCService, name: str) -> str:
    [family] = [f for f in PlcCollector(svc).collect() if f.name == name]
    return family.documentation


class TestRefusalsAreLogged:
    async def test_a_caller_posing_as_the_plc_is_a_warning_with_its_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = plc()
        await svc.process_state(state(step=0))
        with caplog.at_level(logging.INFO, logger="plc_controller.service"):
            result = await svc.send_command(
                fuel_valve=0.4,
                feedwater_valve=0.5,
                steam_valve=0.6,
                source=pb2.CommandSource.SAFETY,
                operator_id="mallory",
            )
        assert not result.accepted
        [record] = [r for r in caplog.records if "reserved" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "mallory" in record.getMessage()

    async def test_refused_resets_mode_changes_and_targets_are_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = plc()
        await svc.process_state(state(step=0, water_level_m=0.4))
        assert svc.mode is RuntimeMode.ESTOP
        with caplog.at_level(logging.INFO, logger="plc_controller"):
            assert not (await svc.reset_emergency_stop("eng")).accepted
            assert not (await svc.set_mode(RuntimeMode.AUTO, "eng")).accepted
            assert not svc.update_setpoints(999.0e5, 4.8, 811.0, "eng").accepted
            assert not svc.set_load_demand(-1.0, "eng").accepted
        refused = [r for r in caplog.records if r.getMessage().startswith("Refused")]
        assert len(refused) == 4
        assert all(r.levelno == logging.INFO for r in refused)


class TestPublisherMetrics:
    def test_the_collector_reports_lost_messages_and_the_broker_link(self) -> None:
        svc, _ = plc()
        publisher = RecordingPublisher()
        publisher.dropped, publisher.connected, publisher.failures = 7, True, 3
        svc._publisher = publisher  # type: ignore[assignment]
        metrics = samples(svc)
        assert metrics["plc_publish_dropped_total"] == 7.0
        assert metrics["plc_mqtt_connected"] == 1.0
        assert metrics["plc_mqtt_connection_failures_total"] == 3.0

    def test_the_warning_counter_says_what_it_counts(self) -> None:
        svc, _ = plc()
        assert documentation(svc, "plc_warnings") == (
            "Scans that raised an interlock warning."
        )
