"""
Integration tests for the live PhysicsService gRPC server.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
import pytest
import pytest_asyncio
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.server import PhysicsServicer


class TestPhysicsGrpc:
    @pytest_asyncio.fixture(autouse=True)
    async def setup_server(self) -> AsyncGenerator[None, None]:
        self.runtime = PhysicsRuntime(
            PhysicsRuntimeConfig(
                speed_factor=100.0,
                dt=1.0,
            )
        )
        await self.runtime.start()

        self.server = grpc.aio.server()
        pb2_grpc.add_PhysicsServiceServicer_to_server(
            PhysicsServicer(self.runtime),
            self.server,
        )
        port = self.server.add_insecure_port("[::]:0")
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"localhost:{port}")
        self.stub = pb2_grpc.PhysicsServiceStub(self.channel)
        yield
        await self.channel.close()
        await self.server.stop(grace=0)
        await self.runtime.stop()

    @pytest.mark.asyncio
    async def test_health_returns_running(self) -> None:
        response = await self.stub.Health(pb2.Empty())
        assert response.status == "running"
        assert response.service == "physics-engine"

    @pytest.mark.asyncio
    async def test_get_system_state_returns_live_snapshot(self) -> None:
        response = await self.stub.GetSystemState(pb2.Empty())
        assert response.boiler.pressure_pa > 0
        assert response.turbine.electrical_power_w >= 0

    @pytest.mark.asyncio
    async def test_apply_control_command_changes_live_state(self) -> None:
        before = await self.stub.GetSystemState(pb2.Empty())

        ack = await self.stub.ApplyControlCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.1,
                feedwater_valve=0.4,
                steam_valve=0.0,
                source=pb2.CommandSource.OPERATOR,
                operator_id="test-operator",
            )
        )
        assert ack.accepted

        await asyncio.sleep(0.2)
        after = await self.stub.GetSystemState(pb2.Empty())
        assert after.turbine.electrical_power_w < before.turbine.electrical_power_w

    @pytest.mark.asyncio
    async def test_stream_system_state_yields_real_updates(self) -> None:
        stream = self.stub.StreamSystemState(pb2.StreamRequest(interval_s=0.0))
        first = await stream.read()
        second = await stream.read()
        stream.cancel()
        assert first.boiler.timestamp_ms > 0
        assert second.boiler.timestamp_ms > first.boiler.timestamp_ms
