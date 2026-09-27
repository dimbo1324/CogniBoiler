"""The storage policy, the subscriber at work, and the historian's entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cogniboiler_pb2 as pb
import pytest
from aiomqtt import MqttError
from historian import __main__ as entry
from historian import storage, subscriber
from historian.stats import report_stats
from historian.storage import (
    StoragePolicy,
    apply_policy,
    downsample_flux,
    ensure_storage,
    flux_string,
)
from historian.subscriber import HistorianSubscriber

POLICY = StoragePolicy(
    org="cogniboiler",
    raw_bucket="sensors",
    aggregate_bucket="sensors_1m",
    raw_retention_days=7,
    aggregate_retention_days=90,
)
DAY_S = 86_400


class Bucket(SimpleNamespace):
    name: str
    retention_rules: list[Any]


class FakeInflux:
    """The parts of the InfluxDB client the storage policy uses."""

    organizations: list[Any] = [SimpleNamespace(id="org-1")]
    buckets: dict[str, Bucket] = {}
    tasks: list[SimpleNamespace] = []
    created_tasks: list[Any] = []
    deleted_tasks: list[str] = []
    updated: list[str] = []
    closed = 0

    def __init__(self, **_: object) -> None:
        return None

    def organizations_api(self) -> FakeInflux:
        return self

    def find_organizations(self, org: str) -> list[Any]:
        return FakeInflux.organizations

    def buckets_api(self) -> FakeInflux:
        return self

    def find_bucket_by_name(self, name: str) -> Bucket | None:
        return FakeInflux.buckets.get(name)

    def create_bucket(
        self, *, bucket_name: str, org_id: str, retention_rules: list[Any]
    ) -> None:
        FakeInflux.buckets[bucket_name] = Bucket(
            name=bucket_name, retention_rules=retention_rules
        )

    def update_bucket(self, bucket: Bucket) -> None:
        FakeInflux.updated.append(bucket.name)

    def tasks_api(self) -> FakeInflux:
        return self

    def find_tasks(self, name: str) -> list[SimpleNamespace]:
        return FakeInflux.tasks

    def delete_task(self, task_id: str) -> None:
        FakeInflux.deleted_tasks.append(task_id)

    def create_task(self, *, task_create_request: Any) -> None:
        FakeInflux.created_tasks.append(task_create_request)

    def close(self) -> None:
        FakeInflux.closed += 1


@pytest.fixture
def influx(monkeypatch: pytest.MonkeyPatch) -> type[FakeInflux]:
    FakeInflux.organizations = [SimpleNamespace(id="org-1")]
    FakeInflux.buckets = {}
    FakeInflux.tasks = []
    FakeInflux.created_tasks = []
    FakeInflux.deleted_tasks = []
    FakeInflux.updated = []
    FakeInflux.closed = 0
    monkeypatch.setattr(storage, "InfluxDBClient", FakeInflux)
    return FakeInflux


def retention(days: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(every_seconds=days * DAY_S)]


class TestStoragePolicy:
    def test_the_downsampling_task_reads_the_raw_and_writes_the_aggregates(
        self,
    ) -> None:
        flux = downsample_flux(POLICY)
        assert 'from(bucket: "sensors")' in flux
        assert 'to(bucket: "sensors_1m", org: "cogniboiler")' in flux
        for measurement in ("boiler_sensors", "turbine_sensors", "plant_status"):
            assert f'r._measurement == "{measurement}"' in flux
        for aggregate in ("mean", "min", "max"):
            assert f'set(key: "agg", value: "{aggregate}")' in flux

    def test_a_bucket_name_with_a_quote_cannot_rewrite_the_task(self) -> None:
        # Bucket and organisation come from the environment: a name with a quote in it
        # would otherwise close the literal and change the task this service installs.
        hostile = 'sensors" |> yield(name: "leak'
        flux = downsample_flux(
            StoragePolicy(
                org="cogniboiler",
                raw_bucket=hostile,
                aggregate_bucket="sensors_1m",
            )
        )
        assert flux_string(hostile) in flux
        # With that one literal taken out, the task is the task it was meant to be.
        skeleton = flux.replace(flux_string(hostile), '"bucket"')
        assert "yield(name:" not in skeleton
        assert skeleton.count("from(bucket:") == 1

    @pytest.mark.parametrize(
        ("value", "literal"),
        [
            ("sensors", '"sensors"'),
            ('a"b', '"a\\"b"'),
            ("a\\b", '"a\\\\b"'),
            ("a${r}", '"a\\${r}"'),
            ("cost $5", '"cost $5"'),
            ("a\nb\rc\td", '"a\\nb\\rc\\td"'),
            ("Kraftwerk Süd", '"Kraftwerk Süd"'),
        ],
    )
    def test_a_flux_literal_escapes_what_flux_would_read(
        self, value: str, literal: str
    ) -> None:
        assert flux_string(value) == literal

    @pytest.mark.parametrize("value", ["a\x01b", "a\x00", "a\x7f", "a\x1bb"])
    def test_other_control_characters_are_refused(self, value: str) -> None:
        # Flux has no \uXXXX escape: json.dumps output broke the task.
        with pytest.raises(ValueError, match="control character"):
            flux_string(value)

    def test_an_interpolation_in_a_bucket_name_stays_text(self) -> None:
        # ${...} inside a Flux string literal is evaluated as an expression.
        flux = downsample_flux(
            StoragePolicy(
                org="cogniboiler", raw_bucket="a${r}", aggregate_bucket="sensors_1m"
            )
        )
        assert 'from(bucket: "a\\${r}")' in flux
        assert '"a${' not in flux

    def test_missing_buckets_and_task_are_created(
        self, influx: type[FakeInflux]
    ) -> None:
        apply_policy("http://influx:8086", "token", POLICY)
        assert {
            name: bucket.retention_rules[0].every_seconds
            for name, bucket in influx.buckets.items()
        } == {"sensors": 7 * DAY_S, "sensors_1m": 90 * DAY_S}
        (task,) = influx.created_tasks
        assert task.flux == downsample_flux(POLICY)
        assert task.status == "active"
        assert influx.closed == 1

    def test_a_wrong_retention_is_corrected(self, influx: type[FakeInflux]) -> None:
        influx.buckets = {
            "sensors": Bucket(name="sensors", retention_rules=retention(30)),
            "sensors_1m": Bucket(name="sensors_1m", retention_rules=retention(90)),
        }
        apply_policy("http://influx:8086", "token", POLICY)
        assert influx.updated == ["sensors"]
        assert influx.buckets["sensors"].retention_rules[0].every_seconds == 7 * DAY_S

    def test_an_identical_active_task_is_kept(self, influx: type[FakeInflux]) -> None:
        influx.tasks = [
            SimpleNamespace(id="t1", flux=downsample_flux(POLICY), status="active")
        ]
        apply_policy("http://influx:8086", "token", POLICY)
        assert influx.created_tasks == []
        assert influx.deleted_tasks == []

    def test_an_outdated_or_inactive_task_is_replaced(
        self, influx: type[FakeInflux]
    ) -> None:
        influx.tasks = [
            SimpleNamespace(id="old", flux="option task = {}", status="active"),
            SimpleNamespace(id="off", flux=downsample_flux(POLICY), status="inactive"),
        ]
        apply_policy("http://influx:8086", "token", POLICY)
        assert influx.deleted_tasks == ["old", "off"]
        assert len(influx.created_tasks) == 1

    def test_an_unknown_organization_fails_and_still_closes_the_client(
        self, influx: type[FakeInflux]
    ) -> None:
        influx.organizations = []
        with pytest.raises(LookupError, match="cogniboiler"):
            apply_policy("http://influx:8086", "token", POLICY)
        assert influx.closed == 1

    async def test_the_policy_is_retried_until_it_applies(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        attempts: list[int] = []

        def flaky(url: str, token: str, policy: StoragePolicy) -> None:
            attempts.append(1)
            if len(attempts) < 4:
                raise ConnectionError("influxdb starting")

        monkeypatch.setattr(storage, "apply_policy", flaky)
        monkeypatch.setattr(storage, "RETRY_DELAY_S", 0.0)
        with caplog.at_level(logging.DEBUG, logger="historian.storage"):
            await ensure_storage("http://influx:8086", "token", POLICY)
        assert len(attempts) == 4
        # Three failures cost one warning; the repeats go to debug.
        levels = [r.levelno for r in caplog.records]
        assert levels.count(logging.WARNING) == 1
        assert levels.count(logging.DEBUG) == 2
        assert "Storage policy applied: sensors 7 d raw, sensors_1m 90 d" in caplog.text


class RecordingWriter:
    def __init__(self) -> None:
        self.batches: list[list[Any]] = []
        self.errors = 0
        self.closed = False

    def write_point(self, point: Any) -> int:
        return self.write_points([point])

    def write_points(self, points: list[Any]) -> int:
        self.batches.append(points)
        return len(points)

    def close(self) -> None:
        self.closed = True

    @property
    def lines(self) -> list[str]:
        return [p.to_line_protocol() for batch in self.batches for p in batch]


def plant(scenario: str, run_id: int, faults: list[str] = []) -> bytes:  # noqa: B006
    return pb.PlantStatusMsg(
        simulation=pb.SimulationStatusMsg(scenario=scenario, run_id=run_id),
        active_faults=[pb.FaultMsg(fault_id=f, label=f) for f in faults],
        timestamp_ms=1_000,
    ).SerializeToString()


class TestSubscriber:
    async def test_boiler_and_turbine_values_carry_the_scenario(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store)  # type: ignore[arg-type]
        await sub._handle_message("sensors/plant", plant("hot_start", 2))
        await sub._handle_message(
            "sensors/boiler",
            pb.BoilerStateMsg(pressure_pa=1.0, timestamp_ms=1_000).SerializeToString(),
        )
        await sub._handle_message(
            "sensors/turbine",
            pb.TurbineStateMsg(steam_flow_kg_s=1.0, timestamp_ms=1).SerializeToString(),
        )
        boiler, turbine = store.lines[1:]
        assert boiler.startswith("boiler_sensors,quality=good,scenario=hot_start ")
        assert turbine.startswith("turbine_sensors,scenario=hot_start ")

    async def test_run_changes_become_simulation_events(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store)  # type: ignore[arg-type]
        await sub._handle_message("sensors/plant", plant("nominal", 1))
        await sub._handle_message("sensors/plant", plant("nominal", 1, ["steam_leak"]))
        await sub._handle_message("sensors/plant", plant("nominal", 1))
        events = [line for line in store.lines if line.startswith("simulation_events")]
        assert [line.split(" ")[0] for line in events] == [
            "simulation_events,kind=fault_injected",
            "simulation_events,kind=fault_cleared",
        ]

    async def test_json_events_availability_and_rejections(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store)  # type: ignore[arg-type]
        at_ms = 1_741_000_000_000
        alarm = {
            "alarm": {"id": 1, "severity": "warning", "parameter": "p", "message": "m"},
            "transition": {"at_ms": at_ms, "to_state": "ACTIVE_UNACK", "actor": "plc"},
        }
        trip = {"kind": "trip", "timestamp_ms": at_ms}
        messages = [
            ("alarms/changes", json.dumps(alarm).encode()),
            ("plc/events", json.dumps(trip).encode()),
            ("status/plc-controller", b"offline"),
            ("alarms/changes", b"{not json"),
            ("alarms/changes", b"[]"),
            ("plc/events", json.dumps({"kind": "trip"}).encode()),
            ("status/plc-controller", b"booting"),
            ("sensors/boiler", b"\xff\x00\xff"),
            ("sensors/system/heartbeat", b"1700000000000"),
            ("sensors/elsewhere", b"x"),
        ]
        for topic, payload in messages:
            await sub._handle_message(topic, payload)
        measurements = [line.split(",")[0] for line in store.lines]
        assert measurements == ["alarm_changes", "plc_events", "service_availability"]
        assert sub.stats == {"received": 10, "stored": 3, "skipped": 7}

    @pytest.mark.parametrize("topic", ["alarms/changes", "plc/events"])
    async def test_deeply_nested_json_is_skipped_not_raised(
        self, topic: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Escaping the handler, it tore down the MQTT session: 5 s without history.
        sub = HistorianSubscriber(RecordingWriter())  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger="historian.subscriber"):
            await sub._handle_message(topic, b"[" * 50_000)
        assert sub.stats["skipped"] == 1
        assert "MQTT error" not in caplog.text

    async def test_an_oversized_json_payload_is_skipped(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store)  # type: ignore[arg-type]
        huge = (
            b'{"kind": "trip", "timestamp_ms": 1741000000000, "pad": "'
            + b"x" * (64 * 1024)
            + b'"}'
        )
        await sub._handle_message("plc/events", huge)
        assert sub.stats["skipped"] == 1
        assert store.lines == []

    async def test_an_integer_too_large_for_a_float_does_not_escape(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store)  # type: ignore[arg-type]
        alarm = {
            "alarm": {"id": 10**400, "severity": "warning", "value": 10**400},
            "transition": {"at_ms": 1_741_000_000_000, "to_state": "ACTIVE_UNACK"},
        }
        await sub._handle_message("alarms/changes", json.dumps(alarm).encode())
        (line,) = store.lines
        assert "alarm_id=0" in line and "value=" not in line

    async def test_any_handler_failure_is_skipped_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        sub = HistorianSubscriber(RecordingWriter())  # type: ignore[arg-type]

        def broken(raw: bytes) -> None:
            raise TypeError("unexpected shape")

        monkeypatch.setitem(sub._handlers, "plc/events", broken)
        with caplog.at_level(logging.WARNING, logger="historian.subscriber"):
            await sub._handle_message("plc/events", b"{}")
        assert sub.stats["skipped"] == 1
        assert "plc/events" in caplog.text and "TypeError" in caplog.text

    async def test_points_are_batched_and_flushed_on_size(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store, batch_size=3, flush_interval_s=3600)  # type: ignore[arg-type]
        for _ in range(4):
            await sub._handle_message("status/historian", b"online")
        assert [len(batch) for batch in store.batches] == [3]
        assert sub.stats["stored"] == 3

    async def test_a_partial_batch_is_flushed_when_messages_stop(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store, batch_size=50, flush_interval_s=0.1)  # type: ignore[arg-type]
        await sub._handle_message("status/historian", b"online")
        assert store.batches == []
        task = asyncio.create_task(sub.flush_periodically())
        try:
            async with asyncio.timeout(5.0):
                while not store.batches:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
        assert sub.stats["stored"] == 1

    async def test_flushes_run_one_at_a_time_and_in_order(self) -> None:
        # flush_periodically and _store_point can both flush; two writes at once
        # raced the writer's counters and could reorder batches.
        release = threading.Event()
        entered = threading.Event()

        class SlowWriter(RecordingWriter):
            def write_points(self, points: list[Any]) -> int:
                if not self.batches:
                    entered.set()
                    release.wait(5.0)
                return super().write_points(points)

        store = SlowWriter()
        sub = HistorianSubscriber(store, batch_size=50, flush_interval_s=3600)  # type: ignore[arg-type]
        await sub._handle_message("status/physics-engine", b"online")
        first = asyncio.create_task(sub.flush())
        await asyncio.to_thread(entered.wait, 5.0)
        await sub._handle_message("status/plc-controller", b"online")
        second = asyncio.create_task(sub.flush())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not second.done()
        assert len(store.batches) == 0
        release.set()
        await asyncio.gather(first, second)
        assert [line.split(" ")[0] for line in store.lines] == [
            "service_availability,service=physics-engine",
            "service_availability,service=plc-controller",
        ]
        assert sub.stats["stored"] == 2

    async def test_an_overdue_flush_happens_on_the_next_point(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store, batch_size=50, flush_interval_s=0.1)  # type: ignore[arg-type]
        sub._last_flush_at -= 1.0
        await sub._handle_message("status/historian", b"online")
        assert [len(batch) for batch in store.batches] == [1]


class FakeBroker:
    hold = False
    connections: list[dict[str, Any]] = []
    subscriptions: list[tuple[str, int]] = []
    deliveries: list[SimpleNamespace] = []

    def __init__(self, **options: Any) -> None:
        FakeBroker.connections.append(options)

    async def __aenter__(self) -> FakeBroker:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def subscribe(self, topic: str, qos: int) -> None:
        FakeBroker.subscriptions.append((topic, qos))

    @property
    def messages(self) -> AsyncIterator[SimpleNamespace]:
        pending, FakeBroker.deliveries = FakeBroker.deliveries, []

        async def deliver() -> AsyncIterator[SimpleNamespace]:
            for message in pending:
                yield message
            if FakeBroker.hold:
                await asyncio.Event().wait()
            raise MqttError("connection lost")

        return deliver()


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> type[FakeBroker]:
    FakeBroker.hold = False
    FakeBroker.connections = []
    FakeBroker.subscriptions = []
    FakeBroker.deliveries = []
    monkeypatch.setattr(subscriber, "Client", FakeBroker)
    monkeypatch.setattr(subscriber, "RECONNECT_DELAY_S", 0.001)
    return FakeBroker


class TestSubscriberSession:
    async def test_a_persistent_session_subscribes_to_every_contract_topic(
        self, broker: type[FakeBroker], caplog: pytest.LogCaptureFixture
    ) -> None:
        broker.deliveries = [
            SimpleNamespace(topic="status/physics-engine", payload=b"online"),
            SimpleNamespace(topic="status/physics-engine", payload="text"),
        ]
        store = RecordingWriter()
        sub = HistorianSubscriber(
            store,  # type: ignore[arg-type]
            "broker",
            1883,
            batch_size=10,
            client_id="historian",
            mqtt_username="historian",
            mqtt_password="pw",
        )
        with caplog.at_level(logging.WARNING, logger="historian.subscriber"):
            task = asyncio.create_task(sub.run())
            try:
                async with asyncio.timeout(5.0):
                    while len(broker.connections) < 2:
                        await asyncio.sleep(0.001)
            finally:
                task.cancel()
        assert broker.subscriptions[:4] == [
            ("sensors/#", 0),
            ("alarms/changes", 1),
            ("plc/events", 1),
            ("status/+", 1),
        ]
        options = broker.connections[0]
        assert (options["clean_session"], options["identifier"]) == (False, "historian")
        assert (
            options["max_queued_incoming_messages"] == subscriber.INCOMING_QUEUE_LIMIT
        )
        assert (options["username"], options["password"]) == ("historian", "pw")
        assert len(store.lines) == 1
        assert sub.stats["skipped"] == 1
        assert sub.connected is False
        assert "Historian: MQTT error" in caplog.text

    async def test_a_subscriber_that_never_connected_is_not_connected(self) -> None:
        # What the container healthcheck reads before the first session opens.
        assert HistorianSubscriber(RecordingWriter()).connected is False  # type: ignore[arg-type]

    async def test_without_a_client_id_the_session_is_clean(
        self, broker: type[FakeBroker]
    ) -> None:
        sub = HistorianSubscriber(RecordingWriter())  # type: ignore[arg-type]
        task = asyncio.create_task(sub.run())
        try:
            async with asyncio.timeout(5.0):
                while not broker.connections:
                    await asyncio.sleep(0.001)
        finally:
            task.cancel()
        assert broker.connections[0]["clean_session"] is None


class TestStats:
    async def test_the_counters_are_logged_and_stored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = RecordingWriter()
        store.errors = 4
        sub = SimpleNamespace(stats={"received": 9, "stored": 7, "skipped": 2})
        with caplog.at_level(logging.INFO, logger="historian.stats"):
            task = asyncio.create_task(report_stats(sub, store, 0.0))  # type: ignore[arg-type]
            try:
                async with asyncio.timeout(5.0):
                    while not store.batches:
                        await asyncio.sleep(0.001)
            finally:
                task.cancel()
        assert store.lines[0].startswith("historian_stats,service=historian ")
        assert "received=9 stored=7 skipped=2 writer_errors=4" in caplog.text


class TestEntryPoint:
    def test_the_command_line_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["historian"])
        args = entry.parse_args()
        assert (args.client_id, args.aggregate_bucket) == ("historian", "sensors_1m")
        assert (args.batch_size, args.metrics_port) == (50, 9103)
        assert (args.raw_retention_days, args.aggregate_retention_days) == (7, 90)
        assert args.influx_timeout_s == 10.0

    async def test_it_wires_the_writer_subscriber_policy_and_stats(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        made: dict[str, Any] = {}
        store = RecordingWriter()

        def new_writer(**options: Any) -> RecordingWriter:
            made["writer"] = options
            return store

        class Sub:
            connected = True
            stats = {"received": 3, "stored": 2, "skipped": 1}

            def __init__(self, **options: Any) -> None:
                made["subscriber"] = options

            async def run(self) -> None:
                await asyncio.Event().wait()

            async def flush_periodically(self) -> None:
                await asyncio.Event().wait()

            async def flush(self) -> None:
                return None

        async def storage_policy(url: str, token: str, policy: StoragePolicy) -> None:
            made["policy"] = policy

        monkeypatch.setattr(entry, "InfluxWriter", new_writer)
        monkeypatch.setattr(entry, "HistorianSubscriber", Sub)
        monkeypatch.setattr(entry, "ensure_storage", storage_policy)
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        monkeypatch.setattr(entry, "STATS_INTERVAL_S", 0.01)
        monkeypatch.setenv("MQTT_PASSWORD", "broker-pass")
        monkeypatch.setattr(
            "sys.argv", ["historian", "--liveness-file", str(tmp_path / "alive")]
        )
        args = entry.parse_args()
        with caplog.at_level(logging.WARNING, logger="historian"):
            task = asyncio.create_task(entry.main(args, ""))
            try:
                async with asyncio.timeout(5.0):
                    while not store.batches or not (tmp_path / "alive").exists():
                        await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert made["writer"]["bucket"] == "sensors"
        assert made["writer"]["timeout_ms"] == 10_000
        assert made["subscriber"]["mqtt_password"] == "broker-pass"
        assert made["subscriber"]["client_id"] == "historian"
        assert made["policy"].aggregate_bucket == "sensors_1m"
        assert store.lines[0].startswith("historian_stats")
        assert "INFLUXDB_TOKEN is empty" in caplog.text

    async def test_stopping_flushes_the_partial_batch_and_closes_the_writer(
        self, broker: type[FakeBroker], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Up to batch_size - 1 points (about 2 s of data) were lost on every stop.
        broker.hold = True
        broker.deliveries = [
            SimpleNamespace(topic="status/physics-engine", payload=b"online")
        ]
        store = RecordingWriter()
        made: list[HistorianSubscriber] = []

        class Recorded(HistorianSubscriber):
            def __init__(self, *args: Any, **options: Any) -> None:
                super().__init__(*args, **options)
                made.append(self)

        async def no_policy(*_: Any) -> None:
            return None

        monkeypatch.setattr(entry, "InfluxWriter", lambda **_: store)
        monkeypatch.setattr(entry, "HistorianSubscriber", Recorded)
        monkeypatch.setattr(entry, "ensure_storage", no_policy)
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        monkeypatch.setattr(entry, "STATS_INTERVAL_S", 3600.0)
        monkeypatch.setattr(
            "sys.argv",
            ["historian", "--flush-interval-s", "3600", "--batch-size", "50"],
        )
        task = asyncio.create_task(entry.main(entry.parse_args(), "token"))
        try:
            async with asyncio.timeout(5.0):
                while not made or made[0].stats["received"] < 1:
                    await asyncio.sleep(0.001)
            assert store.batches == []
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert [line.split(" ")[0] for line in store.lines] == [
            "service_availability,service=physics-engine"
        ]
        assert store.closed is True

    async def test_an_empty_aggregate_bucket_disables_the_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []
        running = asyncio.Event()

        class Sub:
            connected = False
            stats = {"received": 0, "stored": 0, "skipped": 0}

            def __init__(self, **_: Any) -> None:
                return None

            async def run(self) -> None:
                running.set()
                await asyncio.Event().wait()

            async def flush_periodically(self) -> None:
                return None

            async def flush(self) -> None:
                return None

        async def storage_policy(*_: Any) -> None:
            called.append("policy")

        monkeypatch.setattr(entry, "InfluxWriter", lambda **_: RecordingWriter())
        monkeypatch.setattr(entry, "HistorianSubscriber", Sub)
        monkeypatch.setattr(entry, "ensure_storage", storage_policy)
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        args = argparse.Namespace(
            influx_url="http://influx:8086",
            influx_org="cogniboiler",
            influx_bucket="sensors",
            influx_timeout_s=10.0,
            aggregate_bucket="",
            raw_retention_days=7,
            aggregate_retention_days=90,
            mqtt_host="broker",
            mqtt_port=1883,
            batch_size=1,
            flush_interval_s=1.0,
            client_id="",
            liveness_file=None,
            metrics_port=0,
            metrics_host="127.0.0.1",
        )
        task = asyncio.create_task(entry.main(args, "token"))
        # gather has started every task by the time the subscriber runs.
        async with asyncio.timeout(5.0):
            await running.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert called == []
