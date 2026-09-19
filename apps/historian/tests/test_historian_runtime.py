"""The storage policy, the subscriber at work, and the historian's entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cogniboiler_pb2 as pb
import pytest
from aiomqtt import MqttError
from historian import __main__ as entry
from historian import storage, subscriber
from historian.storage import (
    StoragePolicy,
    apply_policy,
    downsample_flux,
    ensure_storage,
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
            if len(attempts) < 3:
                raise ConnectionError("influxdb starting")

        monkeypatch.setattr(storage, "apply_policy", flaky)
        monkeypatch.setattr(storage, "RETRY_DELAY_S", 0.0)
        with caplog.at_level(logging.INFO, logger="historian.storage"):
            await ensure_storage("http://influx:8086", "token", POLICY)
        assert len(attempts) == 3
        assert caplog.text.count("retrying") == 2
        assert "Storage policy applied: sensors 7 d raw, sensors_1m 90 d" in caplog.text


class RecordingWriter:
    def __init__(self) -> None:
        self.single: list[Any] = []
        self.batches: list[list[Any]] = []
        self.errors = 0

    def write_point(self, point: Any) -> None:
        self.single.append(point)

    def write_points(self, points: list[Any]) -> None:
        self.batches.append(points)

    @property
    def lines(self) -> list[str]:
        written = self.single + [p for batch in self.batches for p in batch]
        return [point.to_line_protocol() for point in written]


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
        alarm = {
            "alarm": {"id": 1, "severity": "warning", "parameter": "p", "message": "m"},
            "transition": {"at_ms": 5, "to_state": "ACTIVE_UNACK", "actor": "plc"},
        }
        messages = [
            ("alarms/changes", json.dumps(alarm).encode()),
            ("plc/events", json.dumps({"kind": "trip", "timestamp_ms": 5}).encode()),
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
        assert store.single == []
        task = asyncio.create_task(sub.flush_periodically())
        try:
            async with asyncio.timeout(5.0):
                while not store.single:
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
        assert sub.stats["stored"] == 1

    async def test_an_overdue_flush_happens_on_the_next_point(self) -> None:
        store = RecordingWriter()
        sub = HistorianSubscriber(store, batch_size=50, flush_interval_s=0.1)  # type: ignore[arg-type]
        sub._last_flush_at -= 1.0
        await sub._handle_message("status/historian", b"online")
        assert len(store.single) == 1


class FakeBroker:
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
            raise MqttError("connection lost")

        return deliver()


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> type[FakeBroker]:
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
        assert (options["username"], options["password"]) == ("historian", "pw")
        assert len(store.single) == 1
        assert sub.stats["skipped"] == 1
        assert sub.connected is False
        assert "Historian MQTT error" in caplog.text

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


class TestEntryPoint:
    def test_the_command_line_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["historian"])
        args = entry.parse_args()
        assert (args.client_id, args.aggregate_bucket) == ("historian", "sensors_1m")
        assert (args.batch_size, args.metrics_port) == (50, 9103)
        assert (args.raw_retention_days, args.aggregate_retention_days) == (7, 90)

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
                    while not store.single or not (tmp_path / "alive").exists():
                        await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert made["writer"]["bucket"] == "sensors"
        assert made["subscriber"]["mqtt_password"] == "broker-pass"
        assert made["subscriber"]["client_id"] == "historian"
        assert made["policy"].aggregate_bucket == "sensors_1m"
        assert store.single[0].to_line_protocol().startswith("historian_stats")
        assert "INFLUXDB_TOKEN is empty" in caplog.text

    async def test_an_empty_aggregate_bucket_disables_the_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []

        class Sub:
            connected = False
            stats = {"received": 0, "stored": 0, "skipped": 0}

            def __init__(self, **_: Any) -> None:
                return None

            async def run(self) -> None:
                return None

            async def flush_periodically(self) -> None:
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
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert called == []
