"""Service clients used by the API Gateway."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from datetime import UTC
from typing import Any, Protocol, cast

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
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
    """InfluxDB query settings used by the history endpoint."""

    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "cogniboiler"
    bucket: str = "sensors"


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
        self._channel = grpc.aio.insecure_channel(config.target)
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
        self._channel = grpc.aio.insecure_channel(config.target)
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
        self._channel = grpc.aio.insecure_channel(config.target)
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

# Aggregation windows a history query may use, so neighbouring queries line up.
HISTORY_WINDOWS_S: tuple[int, ...] = (
    1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600,
    43200, 86400,
)  # fmt: skip


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


def build_history_query(
    *,
    bucket: str,
    measurement: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    window_s: int = 0,
    fields: Sequence[str] = (),
) -> str:
    """Flux for one measurement over [start_ms, end_ms], one pivoted row per time.

    With window_s every numeric field is averaged per window; tags are dropped, so
    series split by instrument quality or scenario merge back into one row per window.
    Flux's time(v:) takes integer nanoseconds; InfluxDB rejects float seconds with 400.
    """
    start_ns = start_ms * NANOSECONDS_PER_MILLISECOND
    stop_ns = end_ms * NANOSECONDS_PER_MILLISECOND
    field_filter = ""
    if fields:
        condition = " or ".join(f'r._field == "{field}"' for field in fields)
        field_filter = f"\n  |> filter(fn: (r) => {condition})"
    aggregation = ""
    if window_s > 0:
        aggregation = (
            '\n  |> filter(fn: (r) => types.isType(v: r._value, type: "float"))'
            '\n  |> group(columns: ["_measurement", "_field"])'
            f"\n  |> aggregateWindow(every: {window_s}s, fn: mean, createEmpty: false)"
        )
    return f"""import "types"

from(bucket: "{bucket}")
  |> range(start: time(v: {start_ns}), stop: time(v: {stop_ns}))
  |> filter(fn: (r) => r._measurement == "{measurement}"){field_filter}{aggregation}
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> sort(columns: ["_time"], desc: false)
  |> limit(n: {limit})
"""


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
        """Run a Flux query; each record becomes a dict with timestamp_ms added."""
        tables = self._query_api.query(flux, org=self.config.org)
        rows: list[dict[str, Any]] = []
        for table in tables:
            for record in table.records:
                record = cast(QueryRecordLike, record)
                values = dict(record.values)
                record_time = record.get_time()
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
        flux = build_history_query(
            bucket=self.config.bucket,
            measurement=measurement,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            window_s=window_s,
            fields=fields,
        )
        return self.query_rows(flux)
