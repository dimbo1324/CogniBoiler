"""Service clients used by the API Gateway."""

from __future__ import annotations

from collections.abc import AsyncGenerator
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
class HistorianQueryConfig:
    """InfluxDB query settings used by the history endpoint."""

    url: str = "http://localhost:8086"
    token: str = "cogniboiler-dev-token"
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

    def close(self) -> None: ...


def _new_influx_client(url: str, token: str, org: str) -> InfluxDBClientLike:
    """Create a typed wrapper around the untyped third-party Influx client."""
    client = _InfluxDBClient(url=url, token=token, org=org)
    return cast(InfluxDBClientLike, client)


class PhysicsGatewayClient:
    """Small async wrapper around the generated PhysicsServiceStub."""

    def __init__(self, config: PhysicsGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(config.target)
        self._stub = pb2_grpc.PhysicsServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

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


class PLCGatewayClient:
    """Small async wrapper around the generated PLCServiceStub."""

    def __init__(self, config: PLCGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(config.target)
        self._stub = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

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


NANOSECONDS_PER_MILLISECOND = 1_000_000


def build_history_query(
    *, bucket: str, measurement: str, start_ms: int, end_ms: int, limit: int
) -> str:
    """Flux for one measurement over [start_ms, end_ms], one pivoted row per time.

    Flux's time(v:) takes integer nanoseconds; InfluxDB rejects float seconds with 400.
    """
    start_ns = start_ms * NANOSECONDS_PER_MILLISECOND
    stop_ns = end_ms * NANOSECONDS_PER_MILLISECOND
    return f"""
from(bucket: "{bucket}")
  |> range(start: time(v: {start_ns}), stop: time(v: {stop_ns}))
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
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

    def fetch_history(
        self,
        *,
        measurement: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        flux = build_history_query(
            bucket=self.config.bucket,
            measurement=measurement,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )
        tables = self._query_api.query(flux, org=self.config.org)
        rows: list[dict[str, Any]] = []
        for table in tables:
            for record in table.records:
                record = cast(QueryRecordLike, record)
                values = dict(record.values)
                record_time = record.get_time()
                timestamp_ms = int(record_time.astimezone(UTC).timestamp() * 1000)
                values["timestamp_ms"] = timestamp_ms
                rows.append(values)
        return rows
