"""Read-only gRPC clients for the PLC and alarm projections."""

from __future__ import annotations

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio

TIMEOUT_S = 3.0


class PLCStatusClient:
    """PLCService status for the PLC folder; the OPC UA server never commands the PLC."""

    def __init__(self, target: str) -> None:
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        return await self._stub.GetControlStatus(pb2.Empty(), timeout=TIMEOUT_S)


class AlarmReadClient:
    """Open alarms from AlarmService for the Alarms folder."""

    def __init__(self, target: str) -> None:
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = pb2_grpc.AlarmServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def open_alarms(self, limit: int = 200) -> pb2.AlarmListMsg:
        return await self._stub.ListAlarms(
            pb2.ListAlarmsRequest(open_only=True, limit=limit), timeout=TIMEOUT_S
        )
