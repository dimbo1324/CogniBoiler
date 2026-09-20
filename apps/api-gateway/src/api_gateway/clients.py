"""Service clients used by the API Gateway."""

from __future__ import annotations

import json
import math
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import Any, Protocol, cast

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from cogniboiler_observability import client_interceptors
from cogniboiler_runtime import MILLISECONDS_PER_DAY
from influxdb_client.client.influxdb_client import InfluxDBClient as _InfluxDBClient


@dataclass
class PhysicsGatewayConfig:
    """Connection settings for the live PhysicsService."""

    target: str = "localhost:50052"
    timeout_s: float = 2.0


@dataclass
class PLCGatewayConfig:
    """Connection settings for the live PLCService."""

    target: str = "localhost:50051"
    timeout_s: float = 2.0


@dataclass
class AlarmGatewayConfig:
    """Connection settings for the alert-manager AlarmService."""

    target: str = "localhost:50053"
    timeout_s: float = 3.0


@dataclass
class HistorianQueryConfig:
    """InfluxDB query settings used by the history and KPI endpoints."""

    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "cogniboiler"
    bucket: str = "sensors"
    aggregate_bucket: str = "sensors_1m"
    raw_retention_days: int = 7


class QueryRecordLike(Protocol):
    """Typed subset of an Influx query result row."""

    def get_time(self) -> Any: ...

    values: dict[str, Any]


class QueryApiLike(Protocol):
    """Minimal query API surface used by HistorianQueryClient."""

    def query(self, query: str, org: str) -> list[Any]: ...


class InfluxDBClientLike(Protocol):
    """Minimal client surface needed for history queries."""

    def query_api(self) -> QueryApiLike: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


def _new_influx_client(url: str, token: str, org: str) -> InfluxDBClientLike:
    """Create a typed wrapper around the untyped third-party Influx client."""
    client = _InfluxDBClient(url=url, token=token, org=org)
    return cast(InfluxDBClientLike, client)


class PhysicsGatewayClient:
    """
    Async wrapper around the generated PhysicsServiceStub.

    It reads the plant and controls the simulation. It never sends valve commands:
    those go through the PLC.
    """

    def __init__(self, config: PhysicsGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target, interceptors=client_interceptors()
        )
        self._stub = pb2_grpc.PhysicsServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def get_system_state(self) -> pb2.SystemStateMsg:
        return await self._stub.GetSystemState(
            pb2.Empty(),
            timeout=self.config.timeout_s,
        )

    async def stream_system_state(
        self,
        *,
        interval_s: float = 0.0,
    ) -> AsyncGenerator[pb2.SystemStateMsg]:
        stream = self._stub.StreamSystemState(
            pb2.StreamRequest(interval_s=interval_s),
            timeout=None,
        )
        async for item in stream:
            yield item

    async def get_simulation_status(self) -> pb2.SimulationStatusMsg:
        return await self._stub.GetSimulationStatus(
            pb2.Empty(), timeout=self.config.timeout_s
        )

    async def pause(self, operator_id: str) -> pb2.SimulationAck:
        return await self._stub.PauseSimulation(
            pb2.SimulationControlRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def resume(self, operator_id: str) -> pb2.SimulationAck:
        return await self._stub.ResumeSimulation(
            pb2.SimulationControlRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_speed(
        self, speed_factor: float, operator_id: str
    ) -> pb2.SimulationAck:
        return await self._stub.SetSimulationSpeed(
            pb2.SimulationSpeedRequest(
                speed_factor=speed_factor, operator_id=operator_id
            ),
            timeout=self.config.timeout_s,
        )

    async def step(self, steps: int, operator_id: str) -> pb2.SimulationAck:
        # Stepping publishes every step and may take a while for long requests.
        return await self._stub.StepSimulation(
            pb2.StepRequest(steps=steps, operator_id=operator_id),
            timeout=max(self.config.timeout_s, 60.0),
        )

    async def list_scenarios(self) -> pb2.ScenarioListMsg:
        return await self._stub.ListScenarios(
            pb2.Empty(), timeout=self.config.timeout_s
        )

    async def load_scenario(self, name: str, operator_id: str) -> pb2.SimulationAck:
        # Loading solves an operating point before it answers.
        return await self._stub.LoadScenario(
            pb2.ScenarioRequest(name=name, operator_id=operator_id),
            timeout=max(self.config.timeout_s, 30.0),
        )

    async def inject_fault(self, request: pb2.FaultRequest) -> pb2.FaultAck:
        return await self._stub.InjectFault(request, timeout=self.config.timeout_s)

    async def clear_fault(self, request: pb2.FaultClearRequest) -> pb2.FaultAck:
        return await self._stub.ClearFault(request, timeout=self.config.timeout_s)


class PLCGatewayClient:
    """Small async wrapper around the generated PLCServiceStub."""

    def __init__(self, config: PLCGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target, interceptors=client_interceptors()
        )
        self._stub = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def send_command(
        self,
        command: pb2.ControlCommandMsg,
    ) -> pb2.CommandAck:
        return await self._stub.SendCommand(command, timeout=self.config.timeout_s)

    async def update_setpoints(self, setpoints: pb2.SetpointsMsg) -> pb2.CommandAck:
        return await self._stub.UpdateSetpoints(
            setpoints, timeout=self.config.timeout_s
        )

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        return await self._stub.GetControlStatus(
            pb2.Empty(),
            timeout=self.config.timeout_s,
        )

    async def reset_emergency_stop(self, operator_id: str) -> pb2.CommandAck:
        return await self._stub.ResetEmergencyStop(
            pb2.ResetRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_load_demand(self, load_w: float, operator_id: str) -> pb2.CommandAck:
        return await self._stub.SetLoadDemand(
            pb2.LoadDemandRequest(load_w=load_w, operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_control_mode(self, mode: int, operator_id: str) -> pb2.CommandAck:
        return await self._stub.SetControlMode(
            pb2.ControlModeRequest(mode=mode, operator_id=operator_id),
            timeout=self.config.timeout_s,
        )


class AlarmGatewayClient:
    """Small async wrapper around the generated AlarmServiceStub."""

    def __init__(self, config: AlarmGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target, interceptors=client_interceptors()
        )
        self._stub = pb2_grpc.AlarmServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def list_alarms(self, request: pb2.ListAlarmsRequest) -> pb2.AlarmListMsg:
        return await self._stub.ListAlarms(request, timeout=self.config.timeout_s)

    async def get_alarm(self, alarm_id: int) -> pb2.AlarmDetailMsg:
        return await self._stub.GetAlarm(
            pb2.AlarmRef(alarm_id=alarm_id), timeout=self.config.timeout_s
        )

    async def acknowledge(
        self, alarm_id: int, operator_id: str, comment: str
    ) -> pb2.AcknowledgeResult:
        return await self._stub.AcknowledgeAlarm(
            pb2.AcknowledgeAlarmRequest(
                alarm_id=alarm_id, operator_id=operator_id, comment=comment
            ),
            timeout=self.config.timeout_s,
        )

    async def acknowledge_all(
        self, operator_id: str, comment: str, severity: str
    ) -> pb2.AcknowledgeResult:
        return await self._stub.AcknowledgeAll(
            pb2.AcknowledgeAllRequest(
                operator_id=operator_id, comment=comment, severity=severity
            ),
            timeout=self.config.timeout_s,
        )


# ─── History ──────────────────────────────────────────────────────────────────

NANOSECONDS_PER_MILLISECOND = 1_000_000
AGGREGATE_WINDOW_S = 60
# Raw data answers ranges of up to a day; longer or older ranges read the aggregates.
MAX_RAW_SPAN_MS = MILLISECONDS_PER_DAY

# Aggregation windows a history query may use, so neighbouring queries line up.
HISTORY_WINDOWS_S: tuple[int, ...] = (
    1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600,
    43200, 86400,
)  # fmt: skip

KPI_FIELDS: tuple[str, ...] = (
    "electrical_power_w",
    "fuel_heat_input_w",
    "heat_to_cycle_w",
    "co2_kg_s",
    "nox_ppmv",
    "overall_health_pct",
)


@dataclass(frozen=True, slots=True)
class HistorySource:
    """Where a range is read from: raw telemetry or one-minute aggregates."""

    bucket: str
    aggregated: bool

    @property
    def name(self) -> str:
        return "aggregate" if self.aggregated else "raw"


def history_window_s(start_ms: int, end_ms: int, max_points: int) -> int:
    """
    The smallest standard window that keeps a range within max_points.

    Telemetry arrives once per simulated step — once a second at speed 1, up to fifty
    times a second when the simulation runs fast — so the window, not the sample rate,
    bounds the answer.
    """
    span_s = max(end_ms - start_ms, 1) / 1000.0
    needed = math.ceil(span_s / max(max_points, 1))
    for window in HISTORY_WINDOWS_S:
        if window >= needed:
            return window
    return math.ceil(needed / 86400) * 86400


def choose_source(
    config: HistorianQueryConfig, start_ms: int, end_ms: int, now_ms: int
) -> HistorySource:
    """Raw data for recent ranges of up to a day; aggregates for older or longer ones."""
    raw_horizon_ms = now_ms - config.raw_retention_days * MILLISECONDS_PER_DAY
    if start_ms >= raw_horizon_ms and end_ms - start_ms <= MAX_RAW_SPAN_MS:
        return HistorySource(config.bucket, aggregated=False)
    return HistorySource(config.aggregate_bucket, aggregated=True)


def flux_string(value: str) -> str:
    """A Flux string literal, quoted and escaped.

    Every name that reaches a query — a bucket from the environment, a measurement
    and the field names from a request — goes through this. The routes validate them
    too, but the guarantee belongs with the code that builds the query: a caller
    added later cannot forget a check it never had to make.
    """
    return json.dumps(value, ensure_ascii=False)


def _flux_range(start_ms: int, end_ms: int) -> str:
    start_ns = start_ms * NANOSECONDS_PER_MILLISECOND
    stop_ns = end_ms * NANOSECONDS_PER_MILLISECOND
    return f"range(start: time(v: {start_ns}), stop: time(v: {stop_ns}))"


def build_history_query(
    *,
    bucket: str,
    measurement: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    window_s: int = 0,
    fields: Sequence[str] = (),
    aggregated: bool = False,
) -> str:
    """Flux for one measurement over [start_ms, end_ms], one pivoted row per time.

    With window_s every numeric field is averaged per window; tags are dropped, so
    series split by instrument quality or scenario merge back into one row per window.
    From the aggregate bucket only the one-minute means are read.
    Flux's time(v:) takes integer nanoseconds; InfluxDB rejects float seconds with 400.
    """
    filters = f"\n  |> filter(fn: (r) => r._measurement == {flux_string(measurement)})"
    if aggregated:
        filters += '\n  |> filter(fn: (r) => r.agg == "mean")'
    if fields:
        condition = " or ".join(f"r._field == {flux_string(field)}" for field in fields)
        filters += f"\n  |> filter(fn: (r) => {condition})"
    aggregation = ""
    if window_s > 0:
        aggregation = (
            '\n  |> filter(fn: (r) => types.isType(v: r._value, type: "float"))'
            '\n  |> group(columns: ["_measurement", "_field"])'
            f"\n  |> aggregateWindow(every: {window_s}s, fn: mean, createEmpty: false)"
        )
    return f"""import "types"

from(bucket: {flux_string(bucket)})
  |> {_flux_range(start_ms, end_ms)}{filters}{aggregation}
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> sort(columns: ["_time"], desc: false)
  |> limit(n: {limit})
"""


def build_kpi_query(
    *, bucket: str, start_ms: int, end_ms: int, aggregated: bool
) -> str:
    """
    Flux for the KPI inputs over a range: the mean of each input, the peak NOx, the
    lowest health index and the number of samples. Rows: _field, stat, _value.
    """
    condition = " or ".join(f"r._field == {flux_string(field)}" for field in KPI_FIELDS)

    def source(agg: str) -> str:
        agg_filter = f' and r.agg == "{agg}"' if aggregated else ""
        return (
            f"from(bucket: {flux_string(bucket)})"
            f" |> {_flux_range(start_ms, end_ms)}"
            f' |> filter(fn: (r) => r._measurement == "plant_status"{agg_filter})'
            f" |> filter(fn: (r) => {condition})"
            ' |> group(columns: ["_field"])'
        )

    keep = '|> keep(columns: ["_field", "stat", "_value"])'
    as_float = "|> map(fn: (r) => ({r with _value: float(v: r._value)}))"
    return (
        f'means = {source("mean")} |> mean() |> set(key: "stat", value: "mean") {keep}\n'
        f'peaks = {source("max")} |> filter(fn: (r) => r._field == "nox_ppmv")'
        f' |> max() |> set(key: "stat", value: "max") {keep}\n'
        f'lows = {source("min")} |> filter(fn: (r) => r._field == "overall_health_pct")'
        f' |> min() |> set(key: "stat", value: "min") {keep}\n'
        f"samples = {source('mean')}"
        ' |> filter(fn: (r) => r._field == "electrical_power_w")'
        f' |> count() {as_float} |> set(key: "stat", value: "count") {keep}\n'
        "union(tables: [means, peaks, lows, samples]) |> group()\n"
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


class HistorianQueryClient:
    """Query helper for historical telemetry stored in InfluxDB."""

    def __init__(self, config: HistorianQueryConfig) -> None:
        self.config = config
        self._client = _new_influx_client(config.url, config.token, config.org)
        self._query_api = self._client.query_api()

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        return bool(self._client.ping())

    def query_rows(self, flux: str) -> list[dict[str, Any]]:
        """Run a Flux query; each record becomes a dict, with timestamp_ms if timed."""
        tables = self._query_api.query(flux, org=self.config.org)
        rows: list[dict[str, Any]] = []
        for table in tables:
            for record in table.records:
                values = dict(cast(QueryRecordLike, record).values)
                record_time = values.get("_time")
                if record_time is not None:
                    values["timestamp_ms"] = int(
                        record_time.astimezone(UTC).timestamp() * 1000
                    )
                rows.append(values)
        return rows

    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
        window_s: int = 0,
        fields: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        source = choose_source(self.config, start_ms, end_ms, _now_ms())
        if source.aggregated:
            window_s = max(window_s, AGGREGATE_WINDOW_S)
        flux = build_history_query(
            bucket=source.bucket,
            measurement=measurement,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            window_s=window_s,
            fields=fields,
            aggregated=source.aggregated,
        )
        return self.query_rows(flux)

    def fetch_kpi_inputs(
        self, *, start_ms: int, end_ms: int
    ) -> tuple[HistorySource, dict[tuple[str, str], float]]:
        """KPI inputs over a range, keyed by (field, stat)."""
        source = choose_source(self.config, start_ms, end_ms, _now_ms())
        flux = build_kpi_query(
            bucket=source.bucket,
            start_ms=start_ms,
            end_ms=end_ms,
            aggregated=source.aggregated,
        )
        values: dict[tuple[str, str], float] = {}
        for row in self.query_rows(flux):
            value = row.get("_value")
            if isinstance(value, int | float) and math.isfinite(value):
                values[(str(row.get("_field")), str(row.get("stat")))] = float(value)
        return source, values
