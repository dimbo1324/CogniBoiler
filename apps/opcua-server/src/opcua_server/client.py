"""Read-only gRPC clients for the PLC and alarm projections.

The generated stubs can call every RPC of their service, commands included. Each client
keeps its stub behind a protocol that names only the read it makes, so a command call
here fails the type check, and tests/test_boundaries.py fails on any command RPC named in
the package.
"""

from __future__ import annotations

from typing import Any, Protocol

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from cogniboiler_observability import client_interceptors

TIMEOUT_S = 3.0
OPEN_ALARMS_LISTED = 200


class _PlcStatusReader(Protocol):
    async def GetControlStatus(  # noqa: N802
        self, request: Any, *, timeout: float
    ) -> Any: ...


class _AlarmReader(Protocol):
    async def ListAlarms(self, request: Any, *, timeout: float) -> Any: ...  # noqa: N802


def _channel(target: str) -> grpc.aio.Channel:
    return grpc.aio.insecure_channel(target, interceptors=client_interceptors())


class PLCStatusClient:
    """PLCService status for the PLC folder; the OPC UA server never commands the PLC."""

    def __init__(self, target: str) -> None:
        self._channel = _channel(target)
        self._stub: _PlcStatusReader = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        status: pb2.PLCStatusMsg = await self._stub.GetControlStatus(
            pb2.Empty(), timeout=TIMEOUT_S
        )
        return status


class AlarmReadClient:
    """Open alarms from AlarmService for the Alarms folder."""

    def __init__(self, target: str) -> None:
        self._channel = _channel(target)
        self._stub: _AlarmReader = pb2_grpc.AlarmServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def open_alarms(self, limit: int = OPEN_ALARMS_LISTED) -> pb2.AlarmListMsg:
        alarms: pb2.AlarmListMsg = await self._stub.ListAlarms(
            pb2.ListAlarmsRequest(open_only=True, limit=limit), timeout=TIMEOUT_S
        )
        return alarms
