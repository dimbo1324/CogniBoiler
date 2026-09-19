"""The gateway's upstream clients against in-process gRPC servers and a fake InfluxDB."""

# Servicer methods carry the RPC names of the contract.
# ruff: noqa: N802

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from api_gateway import clients
from api_gateway.clients import (
    AlarmGatewayClient,
    AlarmGatewayConfig,
    HistorianQueryClient,
    HistorianQueryConfig,
    PhysicsGatewayClient,
    PhysicsGatewayConfig,
    PLCGatewayClient,
    PLCGatewayConfig,
    choose_source,
    history_window_s,
)
from cogniboiler_observability import CORRELATION_METADATA_KEY, correlation_scope
from gateway_fakes import alarm, plc_status, simulation_status, system_state

DAY_MS = 86_400_000


def _metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
    return {key: str(value) for key, value in context.invocation_metadata() or ()}


class Physics(pb2_grpc.PhysicsServiceServicer):
    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.metadata: dict[str, str] = {}

    def _ack(self, name: str, request: Any) -> pb2.SimulationAck:
        self.requests.append((name, request))
        return pb2.SimulationAck(accepted=True, status=simulation_status())

    async def Health(self, request: pb2.Empty, context: Any) -> pb2.HealthStatus:
        self.metadata = _metadata(context)
        return pb2.HealthStatus(service="physics-engine", status="running")

    async def GetSystemState(
        self, request: pb2.Empty, context: Any
    ) -> pb2.SystemStateMsg:
        return system_state()

    async def StreamSystemState(
        self, request: pb2.StreamRequest, context: Any
    ) -> AsyncIterator[pb2.SystemStateMsg]:
        self.requests.append(("stream", request))
        for _ in range(3):
            yield system_state()

    async def GetSimulationStatus(
        self, request: pb2.Empty, context: Any
    ) -> pb2.SimulationStatusMsg:
        return simulation_status(scenario="hot_start")

    async def PauseSimulation(self, request: Any, context: Any) -> pb2.SimulationAck:
        return self._ack("pause", request)

    async def ResumeSimulation(self, request: Any, context: Any) -> pb2.SimulationAck:
        return self._ack("resume", request)

    async def SetSimulationSpeed(self, request: Any, context: Any) -> pb2.SimulationAck:
        return self._ack("speed", request)

    async def StepSimulation(self, request: Any, context: Any) -> pb2.SimulationAck:
        return self._ack("step", request)

    async def ListScenarios(
        self, request: pb2.Empty, context: Any
    ) -> pb2.ScenarioListMsg:
        return pb2.ScenarioListMsg(current="nominal")

    async def LoadScenario(self, request: Any, context: Any) -> pb2.SimulationAck:
        return self._ack("scenario", request)

    async def InjectFault(
        self, request: pb2.FaultRequest, context: Any
    ) -> pb2.FaultAck:
        self.requests.append(("inject", request))
        return pb2.FaultAck(accepted=True)

    async def ClearFault(
        self, request: pb2.FaultClearRequest, context: Any
    ) -> pb2.FaultAck:
        self.requests.append(("clear", request))
        return pb2.FaultAck(accepted=True)


class PLC(pb2_grpc.PLCServiceServicer):
    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []

    def _ack(self, name: str, request: Any) -> pb2.CommandAck:
        self.requests.append((name, request))
        return pb2.CommandAck(accepted=True)

    async def Health(self, request: pb2.Empty, context: Any) -> pb2.HealthStatus:
        return pb2.HealthStatus(service="plc-controller")

    async def SendCommand(self, request: Any, context: Any) -> pb2.CommandAck:
        return self._ack("command", request)

    async def UpdateSetpoints(self, request: Any, context: Any) -> pb2.CommandAck:
        return self._ack("setpoints", request)

    async def GetControlStatus(
        self, request: pb2.Empty, context: Any
    ) -> pb2.PLCStatusMsg:
        return plc_status()

    async def ResetEmergencyStop(self, request: Any, context: Any) -> pb2.CommandAck:
        return self._ack("reset", request)

    async def SetLoadDemand(self, request: Any, context: Any) -> pb2.CommandAck:
        return self._ack("load", request)

    async def SetControlMode(self, request: Any, context: Any) -> pb2.CommandAck:
        return self._ack("mode", request)


class Alarms(pb2_grpc.AlarmServiceServicer):
    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []

    async def Health(self, request: pb2.Empty, context: Any) -> pb2.HealthStatus:
        return pb2.HealthStatus(service="alert-manager")

    async def ListAlarms(self, request: Any, context: Any) -> pb2.AlarmListMsg:
        self.requests.append(("list", request))
        return pb2.AlarmListMsg(alarms=[alarm(7)], total=1)

    async def GetAlarm(self, request: pb2.AlarmRef, context: Any) -> pb2.AlarmDetailMsg:
        if request.alarm_id != 7:
            await context.abort(grpc.StatusCode.NOT_FOUND, "no such alarm")
        return pb2.AlarmDetailMsg(alarm=alarm(7))

    async def AcknowledgeAlarm(
        self, request: Any, context: Any
    ) -> pb2.AcknowledgeResult:
        self.requests.append(("ack", request))
        return pb2.AcknowledgeResult(accepted=True)

    async def AcknowledgeAll(self, request: Any, context: Any) -> pb2.AcknowledgeResult:
        self.requests.append(("ack_all", request))
        return pb2.AcknowledgeResult(accepted=True)


class Upstreams(SimpleNamespace):
    physics: Physics
    plc: PLC
    alarms: Alarms
    target: str


@pytest_asyncio.fixture
async def upstreams() -> AsyncIterator[Upstreams]:
    server = grpc.aio.server()
    found = Upstreams(physics=Physics(), plc=PLC(), alarms=Alarms())
    pb2_grpc.add_PhysicsServiceServicer_to_server(found.physics, server)
    pb2_grpc.add_PLCServiceServicer_to_server(found.plc, server)
    pb2_grpc.add_AlarmServiceServicer_to_server(found.alarms, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    found.target = f"127.0.0.1:{port}"
    try:
        yield found
    finally:
        await server.stop(grace=None)


class TestPhysicsClient:
    async def test_every_call_reaches_the_service(self, upstreams: Upstreams) -> None:
        client = PhysicsGatewayClient(PhysicsGatewayConfig(target=upstreams.target))
        try:
            assert (await client.health()).service == "physics-engine"
            state = await client.get_system_state()
            assert state.boiler.pressure_pa == 140.0e5
            assert (await client.get_simulation_status()).scenario == "hot_start"
            assert (await client.list_scenarios()).current == "nominal"
            await client.pause("eng")
            await client.resume("eng")
            await client.set_speed(5.0, "eng")
            await client.step(10, "eng")
            await client.load_scenario("hot_start", "eng")
            await client.inject_fault(pb2.FaultRequest(target="spray"))
            await client.clear_fault(pb2.FaultClearRequest(all=True))
        finally:
            await client.close()
        names = [name for name, _ in upstreams.physics.requests]
        assert names == [
            "pause",
            "resume",
            "speed",
            "step",
            "scenario",
            "inject",
            "clear",
        ]
        requests = dict(upstreams.physics.requests)
        assert requests["pause"].operator_id == "eng"
        assert requests["speed"].speed_factor == 5.0
        assert requests["step"].steps == 10
        assert requests["scenario"].name == "hot_start"
        assert requests["inject"].target == "spray"
        assert requests["clear"].all is True

    async def test_the_state_stream_yields_each_step(
        self, upstreams: Upstreams
    ) -> None:
        client = PhysicsGatewayClient(PhysicsGatewayConfig(target=upstreams.target))
        try:
            received = [
                message async for message in client.stream_system_state(interval_s=0.5)
            ]
        finally:
            await client.close()
        assert len(received) == 3
        assert upstreams.physics.requests[0][1].interval_s == 0.5

    async def test_the_correlation_id_travels_in_the_metadata(
        self, upstreams: Upstreams
    ) -> None:
        client = PhysicsGatewayClient(PhysicsGatewayConfig(target=upstreams.target))
        try:
            with correlation_scope("request-42"):
                await client.health()
        finally:
            await client.close()
        assert upstreams.physics.metadata[CORRELATION_METADATA_KEY] == "request-42"

    async def test_an_unreachable_service_raises_an_rpc_error(self) -> None:
        client = PhysicsGatewayClient(
            PhysicsGatewayConfig(target="127.0.0.1:1", timeout_s=0.5)
        )
        try:
            with pytest.raises(grpc.RpcError):
                await client.health()
        finally:
            await client.close()


class TestPlcClient:
    async def test_every_call_reaches_the_service(self, upstreams: Upstreams) -> None:
        client = PLCGatewayClient(PLCGatewayConfig(target=upstreams.target))
        try:
            assert (await client.health()).service == "plc-controller"
            assert (await client.get_control_status()).load_demand_w == 250e6
            await client.send_command(pb2.ControlCommandMsg(fuel_valve=0.3))
            await client.update_setpoints(pb2.SetpointsMsg(pressure_pa=150e5))
            await client.reset_emergency_stop("eng")
            await client.set_load_demand(120e6, "op")
            await client.set_control_mode(pb2.ControlMode.MANUAL, "op")
        finally:
            await client.close()
        requests = dict(upstreams.plc.requests)
        assert requests["command"].fuel_valve == pytest.approx(0.3)
        assert requests["setpoints"].pressure_pa == 150e5
        assert requests["reset"].operator_id == "eng"
        assert (requests["load"].load_w, requests["load"].operator_id) == (120e6, "op")
        assert requests["mode"].mode == pb2.ControlMode.MANUAL


class TestAlarmClient:
    async def test_every_call_reaches_the_service(self, upstreams: Upstreams) -> None:
        client = AlarmGatewayClient(AlarmGatewayConfig(target=upstreams.target))
        try:
            assert (await client.health()).service == "alert-manager"
            listed = await client.list_alarms(pb2.ListAlarmsRequest(open_only=True))
            assert listed.total == 1
            assert (await client.get_alarm(7)).alarm.alarm_id == 7
            await client.acknowledge(7, "op", "seen")
            await client.acknowledge_all("op", "shift", "warning")
        finally:
            await client.close()
        requests = dict(upstreams.alarms.requests)
        assert requests["list"].open_only is True
        assert (requests["ack"].alarm_id, requests["ack"].comment) == (7, "seen")
        assert requests["ack_all"].severity == "warning"

    async def test_an_unknown_alarm_is_not_found(self, upstreams: Upstreams) -> None:
        client = AlarmGatewayClient(AlarmGatewayConfig(target=upstreams.target))
        try:
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await client.get_alarm(99)
        finally:
            await client.close()
        assert failed.value.code() == grpc.StatusCode.NOT_FOUND


class FakeRecord:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get_time(self) -> Any:
        return self.values.get("_time")


class FakeInflux:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []
        self.tables: list[list[dict[str, Any]]] = []
        self.closed = False

    def query_api(self) -> FakeInflux:
        return self

    def query(self, query: str, org: str) -> list[SimpleNamespace]:
        self.queries.append((query, org))
        return [
            SimpleNamespace(records=[FakeRecord(values) for values in table])
            for table in self.tables
        ]

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def influx(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeInflux]:
    fake = FakeInflux()
    monkeypatch.setattr(clients, "_new_influx_client", lambda url, token, org: fake)
    monkeypatch.setattr(clients, "_now_ms", lambda: 30 * DAY_MS)
    yield fake


CONFIG = HistorianQueryConfig(
    url="http://influx:8086",
    token="t",
    org="cogniboiler",
    bucket="sensors",
    aggregate_bucket="sensors_1m",
    raw_retention_days=7,
)


class TestHistorianQueries:
    def test_rows_carry_their_time_in_epoch_milliseconds(
        self, influx: FakeInflux
    ) -> None:
        influx.tables = [
            [
                {
                    "_time": datetime(2026, 9, 15, 6, 10, tzinfo=UTC),
                    "pressure_pa": 1.0,
                },
                {"_field": "x", "_value": 2.0},
            ]
        ]
        client = HistorianQueryClient(CONFIG)
        rows = client.query_rows('from(bucket: "sensors")')
        assert rows[0]["timestamp_ms"] == int(
            datetime(2026, 9, 15, 6, 10, tzinfo=UTC).timestamp() * 1000
        )
        assert "timestamp_ms" not in rows[1]
        assert influx.queries[0][1] == "cogniboiler"
        assert client.ping() is True
        client.close()
        assert influx.closed

    def test_a_recent_short_range_reads_raw_data(self, influx: FakeInflux) -> None:
        client = HistorianQueryClient(CONFIG)
        client.fetch_history(
            measurement="boiler_sensors",
            start_ms=30 * DAY_MS - 3_600_000,
            end_ms=30 * DAY_MS,
            limit=100,
            window_s=5,
            fields=("pressure_pa",),
        )
        flux = influx.queries[-1][0]
        assert 'from(bucket: "sensors")' in flux
        assert "aggregateWindow(every: 5s" in flux
        assert 'r._field == "pressure_pa"' in flux
        assert 'r.agg == "mean"' not in flux

    def test_an_old_range_reads_the_one_minute_means(self, influx: FakeInflux) -> None:
        client = HistorianQueryClient(CONFIG)
        client.fetch_history(
            measurement="boiler_sensors",
            start_ms=10 * DAY_MS,
            end_ms=10 * DAY_MS + 3_600_000,
            limit=100,
        )
        flux = influx.queries[-1][0]
        assert 'from(bucket: "sensors_1m")' in flux
        assert 'r.agg == "mean"' in flux
        assert "aggregateWindow(every: 60s" in flux

    def test_kpi_inputs_keep_only_finite_numbers(self, influx: FakeInflux) -> None:
        influx.tables = [
            [
                {"_field": "electrical_power_w", "stat": "mean", "_value": 250e6},
                {"_field": "nox_ppmv", "stat": "max", "_value": 61},
                {"_field": "co2_kg_s", "stat": "mean", "_value": math.nan},
                {"_field": "heat_to_cycle_w", "stat": "mean", "_value": "n/a"},
            ]
        ]
        source, values = HistorianQueryClient(CONFIG).fetch_kpi_inputs(
            start_ms=30 * DAY_MS - 900_000, end_ms=30 * DAY_MS
        )
        assert source.name == "raw"
        assert values == {
            ("electrical_power_w", "mean"): 250e6,
            ("nox_ppmv", "max"): 61.0,
        }
        flux = influx.queries[-1][0]
        assert "union(tables: [means, peaks, lows, samples])" in flux

    def test_kpis_of_a_long_range_read_the_aggregates(self, influx: FakeInflux) -> None:
        source, _ = HistorianQueryClient(CONFIG).fetch_kpi_inputs(
            start_ms=20 * DAY_MS, end_ms=30 * DAY_MS
        )
        assert source.name == "aggregate"
        assert 'r.agg == "mean"' in influx.queries[-1][0]


class TestRangePlanning:
    @pytest.mark.parametrize(
        ("span_ms", "max_points", "window"),
        [
            (60_000, 600, 1),
            (900_000, 600, 2),
            (3_600_000, 600, 10),
            (86_400_000, 600, 300),
            (90 * DAY_MS, 10, 86_400 * 9),
        ],
    )
    def test_the_smallest_standard_window_that_fits(
        self, span_ms: int, max_points: int, window: int
    ) -> None:
        assert history_window_s(0, span_ms, max_points) == window

    def test_an_empty_range_needs_the_smallest_window(self) -> None:
        assert history_window_s(5, 5, 100) == 1

    @pytest.mark.parametrize(
        ("start_ms", "end_ms", "aggregated"),
        [
            (29 * DAY_MS, 30 * DAY_MS, False),
            (29 * DAY_MS - 1, 30 * DAY_MS, True),
            (22 * DAY_MS, 22 * DAY_MS + 3_600_000, True),
            (23 * DAY_MS, 23 * DAY_MS + 3_600_000, False),
        ],
    )
    def test_raw_for_recent_days_aggregates_otherwise(
        self, start_ms: int, end_ms: int, aggregated: bool
    ) -> None:
        source = choose_source(CONFIG, start_ms, end_ms, now_ms=30 * DAY_MS)
        assert source.aggregated is aggregated
        assert source.bucket == ("sensors_1m" if aggregated else "sensors")
