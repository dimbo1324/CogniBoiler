"""Points the historian writes besides boiler and turbine values, and the writer."""

from __future__ import annotations

import logging
import re
from typing import Any

import cogniboiler_pb2 as pb
import pytest
from cogniboiler_runtime import now_ms
from historian import points, writer
from historian.points import RunLabels
from historian.writer import InfluxWriter, add_numeric_fields, new_point

TS_MS = 1_741_000_000_000
DAY_MS = 86_400_000


def line(point: Any) -> str:
    text: str = point.to_line_protocol()
    return text


def fields(point: Any) -> dict[str, float | str]:
    """Field values of a point, parsed back from its line protocol."""
    tokens = line(point).split(" ")
    body = tokens[1:-1] if re.fullmatch(r"\d+", tokens[-1]) else tokens[1:]
    found: dict[str, float | str] = {}
    pattern = r'(\w+)=("(?:[^"\\]|\\.)*"|[^,]+)'
    for name, raw in re.findall(pattern, " ".join(body)):
        if raw.startswith('"'):
            found[name] = raw[1:-1].replace('\\"', '"')
        else:
            found[name] = float(raw.rstrip("i"))
    return found


def tags(point: Any) -> dict[str, str]:
    head = line(point).split(" ")[0]
    return dict(item.split("=", 1) for item in head.split(",")[1:])


def plant_status(**overrides: Any) -> pb.PlantStatusMsg:
    values: dict[str, Any] = {
        "emissions": pb.EmissionsMsg(co2_kg_s=35.0, nox_ppmv=45.0),
        "condenser": pb.CondenserMsg(backpressure_pa=7000.0, loading=0.9),
        "health": pb.EquipmentHealthMsg(
            overall_health_pct=97.5, maintenance_alarm=True
        ),
        "actuators": pb.ActuatorStateMsg(fuel_valve_command=0.6),
        "performance": pb.PerformanceMsg(net_efficiency=0.4),
        "simulation": pb.SimulationStatusMsg(
            scenario="nominal", run_id=3, speed_factor=10.0, simulation_time_s=60.0
        ),
        "active_faults": [
            pb.FaultMsg(fault_id="b", label="steam_leak"),
            pb.FaultMsg(fault_id="a", label="burner_fouling"),
        ],
        "sensors": [
            pb.SensorStatusMsg(sensor_id="p", quality=pb.SensorQuality.UNCERTAIN),
            pb.SensorStatusMsg(sensor_id="l", quality=pb.SensorQuality.BAD),
            pb.SensorStatusMsg(sensor_id="t", quality=pb.SensorQuality.BAD),
        ],
        "timestamp_ms": TS_MS,
    }
    values.update(overrides)
    return pb.PlantStatusMsg(**values)


class TestPlantPoint:
    def test_every_group_of_the_status_is_a_field(self) -> None:
        point = points.build_plant_point(plant_status())
        found = fields(point)
        assert tags(point) == {"scenario": "nominal"}
        assert found["co2_kg_s"] == 35.0
        assert found["condenser_backpressure_pa"] == 7000.0
        assert found["condenser_loading"] == 0.9
        assert found["overall_health_pct"] == 97.5
        assert found["maintenance_alarm"] == 1.0
        assert found["maintenance_critical"] == 0.0
        assert found["fuel_valve_command"] == 0.6
        assert found["net_efficiency"] == 0.4
        assert (found["speed_factor"], found["run_id"]) == (10.0, 3.0)
        assert found["paused"] == 0.0
        assert found["fault_count"] == 2.0
        assert (found["uncertain_sensor_count"], found["bad_sensor_count"]) == (
            1.0,
            2.0,
        )
        assert found["fault_labels"] == "burner_fouling,steam_leak"
        assert line(point).endswith(str(TS_MS * 1_000_000))

    def test_a_paused_run_without_a_scenario(self) -> None:
        point = points.build_plant_point(
            plant_status(
                simulation=pb.SimulationStatusMsg(
                    run_state=pb.SimulationRunState.SIMULATION_PAUSED
                )
            )
        )
        assert tags(point) == {}
        assert fields(point)["paused"] == 1.0

    def test_a_status_without_a_timestamp_is_stamped_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(points, "now_ms", lambda: 5_000)
        point = points.build_plant_point(plant_status(timestamp_ms=0))
        assert line(point).endswith(" 5000000000")

    @pytest.mark.parametrize("timestamp_ms", [0, 1_000, 10**16, "far future"], ids=str)
    def test_an_implausible_status_time_is_stamped_now(
        self, monkeypatch: pytest.MonkeyPatch, timestamp_ms: int | str
    ) -> None:
        value = now_ms() + 2 * DAY_MS if timestamp_ms == "far future" else timestamp_ms
        monkeypatch.setattr(points, "now_ms", lambda: TS_MS)
        point = points.build_plant_point(plant_status(timestamp_ms=value))
        assert line(point).endswith(f" {TS_MS * 1_000_000}")


class TestTelemetryTime:
    @pytest.mark.parametrize("timestamp_ms", [0, 1, 10**16], ids=str)
    def test_boiler_and_turbine_follow_the_plant_status_rule(
        self, monkeypatch: pytest.MonkeyPatch, timestamp_ms: int
    ) -> None:
        # One rule for protobuf telemetry: a time InfluxDB cannot place (or would
        # file in 1970, outside retention) is replaced by the receive time.
        monkeypatch.setattr(writer, "now_ms", lambda: TS_MS)
        boiler = writer.build_boiler_point(pb.BoilerStateMsg(timestamp_ms=timestamp_ms))
        turbine = writer.build_turbine_point(
            pb.TurbineStateMsg(timestamp_ms=timestamp_ms)
        )
        assert line(boiler).endswith(f" {TS_MS * 1_000_000}")
        assert line(turbine).endswith(f" {TS_MS * 1_000_000}")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (TS_MS, TS_MS),
            (946_684_800_000, 946_684_800_000),
            (946_684_799_999, None),
            (-1, None),
            (0, None),
            (10**16, None),
            (10**400, None),
            (True, None),
            (1.7e12, None),
            ("1741000000000", None),
        ],
        ids=str,
    )
    def test_the_plausible_window(self, value: object, expected: int | None) -> None:
        assert writer.plausible_timestamp_ms(value) == expected

    def test_a_day_ahead_is_the_limit(self) -> None:
        assert writer.plausible_timestamp_ms(now_ms() + DAY_MS - 60_000) is not None
        assert writer.plausible_timestamp_ms(now_ms() + DAY_MS + 60_000) is None


class TestRunLabels:
    def test_the_first_status_only_establishes_the_state(self) -> None:
        labels = RunLabels()
        assert labels.update(plant_status()) == []
        assert (labels.scenario, labels.run_id) == ("nominal", 3)

    def test_a_new_run_is_a_scenario_load_with_its_faults(self) -> None:
        labels = RunLabels()
        labels.update(plant_status())
        events = labels.update(
            plant_status(
                simulation=pb.SimulationStatusMsg(scenario="hot_start", run_id=4),
                active_faults=[pb.FaultMsg(fault_id="c", label="valve_stuck:spray")],
            )
        )
        assert [(e.kind, e.label) for e in events] == [
            ("scenario_loaded", "hot_start"),
            ("fault_injected", "valve_stuck:spray"),
        ]
        assert events[0].text == "Scenario hot_start loaded (run 4)"

    def test_faults_injected_and_cleared_within_a_run(self) -> None:
        labels = RunLabels()
        labels.update(plant_status())
        events = labels.update(
            plant_status(
                active_faults=[
                    pb.FaultMsg(fault_id="b", label="steam_leak"),
                    pb.FaultMsg(fault_id="c", label="sensor_drift:drum_level"),
                ]
            )
        )
        assert sorted((e.kind, e.label) for e in events) == [
            ("fault_cleared", "burner_fouling"),
            ("fault_injected", "sensor_drift:drum_level"),
        ]
        assert all(e.run_id == 3 and e.timestamp_ms == TS_MS for e in events)

    def test_an_unchanged_status_has_no_events(self) -> None:
        labels = RunLabels()
        labels.update(plant_status())
        assert labels.update(plant_status()) == []

    def test_an_event_point_is_an_annotation(self) -> None:
        event = points.SimulationEvent(
            "fault_injected", "nominal", 3, "steam_leak", "Fault injected", TS_MS
        )
        point = points.build_simulation_event_point(event)
        assert tags(point) == {"kind": "fault_injected"}
        found = fields(point)
        assert (found["label"], found["run_id"]) == ("steam_leak", 3.0)
        assert found["text"] == "Fault injected"


def alarm_change(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "alarm": {
            "id": 7,
            "severity": "critical",
            "parameter": "water_level_m",
            "message": "Drum level low-low",
            "value": 3.1,
            "threshold": 3.5,
        },
        "transition": {"at_ms": TS_MS, "to_state": "ACTIVE_UNACK", "actor": "plc"},
    }
    payload.update(overrides)
    return payload


class TestAlarmAndPlcPoints:
    def test_an_alarm_change(self) -> None:
        point = points.build_alarm_change_point(alarm_change())
        assert point is not None
        assert tags(point) == {
            "parameter": "water_level_m",
            "severity": "critical",
            "state": "ACTIVE_UNACK",
        }
        found = fields(point)
        assert (found["alarm_id"], found["value"], found["threshold"]) == (
            7.0,
            3.1,
            3.5,
        )
        assert found["text"] == "critical ACTIVE_UNACK: Drum level low-low"

    @pytest.mark.parametrize(
        "payload",
        [
            {"alarm": {}},
            {"alarm": [], "transition": {"at_ms": 1}},
            alarm_change(transition={"at_ms": "soon"}),
            alarm_change(transition={"at_ms": True}),
        ],
    )
    def test_an_incomplete_alarm_change_is_dropped(
        self, payload: dict[str, Any]
    ) -> None:
        assert points.build_alarm_change_point(payload) is None

    @pytest.mark.parametrize("at_ms", [-1, 0, 5, 10**16, 10**400], ids=str)
    def test_an_alarm_change_at_an_implausible_time_is_dropped(
        self, at_ms: int
    ) -> None:
        payload = alarm_change(transition={"at_ms": at_ms, "to_state": "CLEARED"})
        assert points.build_alarm_change_point(payload) is None

    @pytest.mark.parametrize("timestamp_ms", [-1, 0, 5, 10**16], ids=str)
    def test_a_plc_event_at_an_implausible_time_is_dropped(
        self, timestamp_ms: int
    ) -> None:
        payload = {"kind": "trip", "timestamp_ms": timestamp_ms}
        assert points.build_plc_event_point(payload) is None

    def test_numbers_too_large_for_a_float_are_left_out(self) -> None:
        point = points.build_alarm_change_point(
            alarm_change(alarm={"id": 10**400, "value": 10**400, "threshold": 1})
        )
        assert point is not None
        found = fields(point)
        assert found["alarm_id"] == 0.0
        assert "value" not in found and found["threshold"] == 1.0

    def test_missing_names_and_non_finite_numbers(self) -> None:
        point = points.build_alarm_change_point(
            alarm_change(
                alarm={"id": "x", "value": float("nan"), "threshold": True},
                transition={"at_ms": TS_MS},
            )
        )
        assert point is not None
        assert tags(point) == {
            "parameter": "unknown",
            "severity": "unknown",
            "state": "unknown",
        }
        found = fields(point)
        assert found["alarm_id"] == 0.0
        assert "value" not in found and "threshold" not in found

    def test_a_plc_event_names_its_operator(self) -> None:
        point = points.build_plc_event_point(
            {
                "kind": "mode_change",
                "timestamp_ms": TS_MS,
                "operator_id": "operator1",
                "detail": {"to": "manual", "from": "auto"},
            }
        )
        assert point is not None
        assert tags(point) == {"kind": "mode_change"}
        found = fields(point)
        assert found["text"] == "PLC mode_change by operator1"
        # Keys sorted; quotes escaped by the line protocol.
        assert found["detail"] == '{"from": "auto", "to": "manual"}'

    def test_an_automatic_plc_event(self) -> None:
        point = points.build_plc_event_point({"kind": "trip", "timestamp_ms": TS_MS})
        assert point is not None
        assert fields(point)["text"] == "PLC trip"

    @pytest.mark.parametrize(
        "payload",
        [
            {"timestamp_ms": TS_MS},
            {"kind": 3, "timestamp_ms": TS_MS},
            {"kind": "trip", "timestamp_ms": True},
            {"kind": "trip", "timestamp_ms": "now"},
        ],
    )
    def test_an_incomplete_plc_event_is_dropped(self, payload: dict[str, Any]) -> None:
        assert points.build_plc_event_point(payload) is None


class TestAvailabilityAndStats:
    @pytest.mark.parametrize(
        ("payload", "online"), [(b"online", 1.0), (b" OFFLINE\n", 0.0)]
    )
    def test_the_retained_status_of_a_service(
        self, payload: bytes, online: float
    ) -> None:
        point = points.build_availability_point("physics-engine", payload)
        assert point is not None
        assert tags(point) == {"service": "physics-engine"}
        assert fields(point)["online"] == online

    def test_any_other_status_is_dropped(self) -> None:
        assert points.build_availability_point("plc", b"starting") is None

    def test_the_historian_counts_itself(self) -> None:
        point = points.build_stats_point({"received": 10, "stored": 8}, writer_errors=2)
        assert tags(point) == {"service": "historian"}
        assert fields(point) == {"received": 10.0, "stored": 8.0, "writer_errors": 2.0}


class TestNumericFields:
    def test_bools_are_ones_and_zeros_and_messages_are_skipped(self) -> None:
        message = pb.EquipmentHealthMsg(turbine_hours=10.0, maintenance_alarm=True)
        found = fields(add_numeric_fields(new_point("m"), message, prefix="x_"))
        assert found["x_turbine_hours"] == 10.0
        assert found["x_maintenance_alarm"] == 1.0

    def test_the_timestamp_and_quality_are_not_fields(self) -> None:
        message = pb.BoilerStateMsg(pressure_pa=1.0, timestamp_ms=5, quality=2)
        found = fields(add_numeric_fields(new_point("m"), message))
        assert "timestamp_ms" not in found and "quality" not in found


class FakeWriteApi:
    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []
        self.fail = False

    def write(self, *, bucket: str, record: Any) -> None:
        if self.fail:
            raise ConnectionError("influxdb refused")
        self.writes.append((bucket, record))


class FakeInfluxClient:
    def __init__(self) -> None:
        self.api = FakeWriteApi()
        self.closed = False

    def write_api(self, *, write_options: object) -> FakeWriteApi:
        return self.api

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def influx(monkeypatch: pytest.MonkeyPatch) -> FakeInfluxClient:
    client = FakeInfluxClient()
    monkeypatch.setattr(writer, "_new_client", lambda url, token, org: client)
    return client


class TestWriter:
    def test_points_are_written_to_the_bucket_and_counted(
        self, influx: FakeInfluxClient
    ) -> None:
        store = InfluxWriter("http://influx:8086", "t", "org", "sensors")
        assert store.write_point(new_point("a")) == 1
        assert store.write_points([new_point("b"), new_point("c")]) == 2
        assert store.write_points([]) == 0
        assert [bucket for bucket, _ in influx.api.writes] == ["sensors", "sensors"]
        assert (store.written, store.errors) == (3, 0)
        store.close()
        assert influx.closed

    def test_failures_are_counted_and_logged_not_raised(
        self, influx: FakeInfluxClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        influx.api.fail = True
        store = InfluxWriter("http://influx:8086", "t", "org", "sensors")
        with caplog.at_level(logging.WARNING, logger="historian.writer"):
            assert store.write_point(new_point("a")) == 0
            assert store.write_points([new_point("b"), new_point("c")]) == 0
        assert (store.written, store.errors) == (0, 3)
        assert "InfluxDB write" in caplog.text

    def test_an_outage_is_one_warning_and_one_recovery_line(
        self, influx: FakeInfluxClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        # One line per failed write was ~1800 lines an hour with InfluxDB down.
        store = InfluxWriter("http://influx:8086", "t", "org", "sensors")
        influx.api.fail = True
        with caplog.at_level(logging.DEBUG, logger="historian.writer"):
            store.write_point(new_point("a"))
            store.write_points([new_point("b"), new_point("c")])
            store.write_point(new_point("d"))
            influx.api.fail = False
            store.write_point(new_point("e"))
            store.write_point(new_point("f"))
        levels = [r.levelno for r in caplog.records]
        assert levels.count(logging.WARNING) == 1
        assert levels.count(logging.DEBUG) == 2
        recoveries = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        assert recoveries == ["InfluxDB writes succeed again (4 points lost)"]

    def test_a_second_outage_warns_again(
        self, influx: FakeInfluxClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = InfluxWriter("http://influx:8086", "t", "org", "sensors")
        with caplog.at_level(logging.WARNING, logger="historian.writer"):
            for fail in (True, False, True):
                influx.api.fail = fail
                store.write_point(new_point("a"))
        assert [r.levelno for r in caplog.records].count(logging.WARNING) == 2
