"""Minimal PLC gRPC client used by OPC UA methods."""

from __future__ import annotations

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio


class PLCControlClient:
    """Small async wrapper for OPC UA method callbacks."""

    def __init__(self, target: str) -> None:
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def apply_manual_command(
        self,
        *,
        fuel_valve: float,
        feedwater_valve: float,
        steam_valve: float,
    ) -> bool:
        ack = await self._stub.SendCommand(
            pb2.ControlCommandMsg(
                fuel_valve=fuel_valve,
                feedwater_valve=feedwater_valve,
                steam_valve=steam_valve,
                source=pb2.CommandSource.OPERATOR,
                operator_id="opcua-method",
            )
        )
        return bool(ack.accepted)

    async def reset_estop(self) -> bool:
        ack = await self._stub.ResetEmergencyStop(
            pb2.ResetRequest(operator_id="opcua-method")
        )
        return bool(ack.accepted)
