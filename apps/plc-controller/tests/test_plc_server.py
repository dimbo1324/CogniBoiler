"""
Tests for PLCService business logic and gRPC server.

Split into two classes:
  TestPLCService  — pure unit tests (no network, no gRPC)
  TestPLCGrpc     — integration tests against a real in-process gRPC server
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import replace

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from physics_engine.models import BoilerParameters
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.server import PhysicsServicer
from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.server import PLCServicer
from plc_controller.service import (
    PRESSURE_SETPOINT_MAX_PA,
    PRESSURE_SETPOINT_MIN_PA,
    PLCService,
)

# ─── Pure unit tests ──────────────────────────────────────────────────────────


class TestPLCService:
    """Unit tests for PLCService — no gRPC needed."""

    def setup_method(self) -> None:
        self.svc = PLCService()

    # ── Command validation ────────────────────────────────────────────────────

    def test_valid_command_accepted(self) -> None:
        result = self.svc.validate_command(0.5, 0.5, 0.5)
        assert result.accepted

    def test_fuel_valve_above_max_rejected(self) -> None:
        result = self.svc.validate_command(1.1, 0.5, 0.5)
        assert not result.accepted
        assert "fuel_valve" in result.reason

    def test_feedwater_valve_negative_rejected(self) -> None:
        result = self.svc.validate_command(0.5, -0.1, 0.5)
        assert not result.accepted
        assert "feedwater_valve" in result.reason

    def test_steam_valve_boundary_zero_accepted(self) -> None:
        result = self.svc.validate_command(0.5, 0.5, 0.0)
        assert result.accepted

    def test_steam_valve_boundary_one_accepted(self) -> None:
        result = self.svc.validate_command(0.5, 0.5, 1.0)
        assert result.accepted

    def test_rejected_commands_counted(self) -> None:
        self.svc.validate_command(1.5, 0.5, 0.5)  # rejected
        self.svc.validate_command(0.5, 0.5, 0.5)  # accepted
        assert self.svc.stats["commands_received"] == 2
        assert self.svc.stats["commands_rejected"] == 1

    # ── Setpoints ────────────────────────────────────────────────────────────

    def test_default_setpoints_are_nominal(self) -> None:
        sp = self.svc.get_setpoints()
        assert sp.pressure_pa == pytest.approx(140.0e5)
        assert sp.water_level_m == pytest.approx(4.8)

    def test_valid_setpoints_accepted(self) -> None:
        result = self.svc.update_setpoints(
            pressure_pa=130.0e5,
            water_level_m=5.0,
            steam_temp_k=800.0,
        )
        assert result.accepted
        sp = self.svc.get_setpoints()
        assert sp.pressure_pa == pytest.approx(130.0e5)

    def test_pressure_setpoint_too_high_rejected(self) -> None:
        result = self.svc.update_setpoints(
            pressure_pa=PRESSURE_SETPOINT_MAX_PA + 1.0,
            water_level_m=4.8,
            steam_temp_k=811.0,
        )
        assert not result.accepted
        assert "Pressure" in result.reason

    def test_pressure_setpoint_too_low_rejected(self) -> None:
        result = self.svc.update_setpoints(
            pressure_pa=PRESSURE_SETPOINT_MIN_PA - 1.0,
            water_level_m=4.8,
            steam_temp_k=811.0,
        )
        assert not result.accepted

    def test_rejected_setpoint_does_not_change_state(self) -> None:
        original = self.svc.get_setpoints().pressure_pa
        self.svc.update_setpoints(999.0e5, 4.8, 811.0)  # rejected
        assert self.svc.get_setpoints().pressure_pa == pytest.approx(original)

    def test_uptime_increases(self) -> None:
        t0 = self.svc.uptime_seconds
        time.sleep(0.05)
        assert self.svc.uptime_seconds > t0


# ─── gRPC integration tests ───────────────────────────────────────────────────


class TestPLCGrpc:
    """
    Integration tests using grpc.aio in-process server.

    Each test spins up a real server on a free OS port and tears it down.
    This verifies the full gRPC transport layer end-to-end.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup_server(self) -> AsyncGenerator[None]:
        """Start PhysicsService and PLCService on free OS ports."""
        # The plant starts paused and advances only when a test steps it, so no test
        # depends on how fast this machine runs the plant (debt Д2).
        self.physics_runtime = PhysicsRuntime(
            PhysicsRuntimeConfig(dt=1.0, start_paused=True)
        )
        await self.physics_runtime.start()

        self.physics_server = grpc.aio.server()
        pb2_grpc.add_PhysicsServiceServicer_to_server(
            PhysicsServicer(self.physics_runtime),
            self.physics_server,
        )
        physics_port = self.physics_server.add_insecure_port("[::]:0")
        await self.physics_server.start()
        self.physics_channel = grpc.aio.insecure_channel(f"localhost:{physics_port}")
        self.physics_stub = pb2_grpc.PhysicsServiceStub(self.physics_channel)

        self.plc_service = PLCService(
            physics_client=PhysicsClient(
                PhysicsClientConfig(target=f"localhost:{physics_port}")
            ),
            control_interval_s=0.05,
            enable_alert_publishing=False,
        )
        await self.plc_service.start()
        self.server = grpc.aio.server()
        pb2_grpc.add_PLCServiceServicer_to_server(
            PLCServicer(self.plc_service),
            self.server,
        )
        port = self.server.add_insecure_port("[::]:0")  # OS picks free port
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"localhost:{port}")
        self.stub = pb2_grpc.PLCServiceStub(self.channel)
        yield
        await self.plc_service.close()
        await self.channel.close()
        await self.physics_channel.close()
        await self.server.stop(grace=0)
        await self.physics_server.stop(grace=0)
        await self.physics_runtime.stop()

    @pytest.mark.asyncio
    async def test_health_returns_running(self) -> None:
        response = await self.stub.Health(pb2.Empty())
        assert response.status == "running"
        assert response.service == "plc-controller"

    @pytest.mark.asyncio
    async def test_health_version_not_empty(self) -> None:
        response = await self.stub.Health(pb2.Empty())
        assert response.version != ""

    @pytest.mark.asyncio
    async def test_send_valid_command_accepted(self) -> None:
        cmd = pb2.ControlCommandMsg(
            fuel_valve=0.6,
            feedwater_valve=0.5,
            steam_valve=0.55,
            source=pb2.CommandSource.OPERATOR,
        )
        ack = await self.stub.SendCommand(cmd)
        assert ack.accepted

    @pytest.mark.asyncio
    async def test_send_invalid_command_rejected(self) -> None:
        cmd = pb2.ControlCommandMsg(
            fuel_valve=1.5,  # invalid
            feedwater_valve=0.5,
            steam_valve=0.5,
        )
        ack = await self.stub.SendCommand(cmd)
        assert not ack.accepted
        assert "fuel_valve" in ack.reason

    @pytest.mark.asyncio
    async def test_plc_command_reaches_live_physics_runtime(self) -> None:
        before = await self.physics_stub.GetSystemState(pb2.Empty())
        ack = await self.stub.SendCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.1,
                feedwater_valve=0.4,
                steam_valve=0.0,
                source=pb2.CommandSource.OPERATOR,
                operator_id="operator-42",
            )
        )
        assert ack.accepted

        await self.physics_runtime.step(60)
        after = await self.physics_stub.GetSystemState(pb2.Empty())
        assert after.turbine.electrical_power_w < before.turbine.electrical_power_w

    @pytest.mark.asyncio
    async def test_get_setpoints_returns_defaults(self) -> None:
        sp = await self.stub.GetSetpoints(pb2.Empty())
        assert sp.pressure_pa == pytest.approx(140.0e5)
        assert sp.water_level_m == pytest.approx(4.8)

    @pytest.mark.asyncio
    async def test_update_setpoints_accepted(self) -> None:
        new_sp = pb2.SetpointsMsg(
            pressure_pa=130.0e5,
            water_level_m=5.0,
            steam_temp_k=800.0,
        )
        ack = await self.stub.UpdateSetpoints(new_sp)
        assert ack.accepted

    @pytest.mark.asyncio
    async def test_update_setpoints_reflected_in_get(self) -> None:
        await self.stub.UpdateSetpoints(
            pb2.SetpointsMsg(
                pressure_pa=120.0e5,
                water_level_m=5.5,
                steam_temp_k=790.0,
            )
        )
        sp = await self.stub.GetSetpoints(pb2.Empty())
        assert sp.pressure_pa == pytest.approx(120.0e5)

    @pytest.mark.asyncio
    async def test_update_invalid_setpoints_rejected(self) -> None:
        ack = await self.stub.UpdateSetpoints(
            pb2.SetpointsMsg(
                pressure_pa=999.0e5,  # way over limit
                water_level_m=4.8,
                steam_temp_k=811.0,
            )
        )
        assert not ack.accepted

    @pytest.mark.asyncio
    async def test_ack_has_timestamp(self) -> None:
        ack = await self.stub.SendCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.5,
                feedwater_valve=0.5,
                steam_valve=0.5,
            )
        )
        assert ack.timestamp_ms > 0

    @pytest.mark.asyncio
    async def test_command_ack_timestamp_is_recent(self) -> None:
        before_ms = int(time.time() * 1000)
        ack = await self.stub.SendCommand(
            pb2.ControlCommandMsg(
                fuel_valve=0.5,
                feedwater_valve=0.5,
                steam_valve=0.5,
            )
        )
        after_ms = int(time.time() * 1000)
        assert before_ms <= ack.timestamp_ms <= after_ms + 100

    @pytest.mark.asyncio
    async def test_auto_control_holds_nominal_state_for_ten_minutes_simulated(
        self,
    ) -> None:
        # Lockstep over gRPC: one plant step, then the PLC scan of exactly that step, so
        # the outcome does not depend on how fast this machine runs the plant (debt Д2).
        while self.physics_runtime.snapshot.simulation_time_s < 600.0:
            status = await self.physics_runtime.step(1)
            deadline = time.monotonic() + 5.0
            while self.plc_service.last_scanned_step < status.step_count:
                assert time.monotonic() < deadline, (
                    f"PLC did not scan step {status.step_count} within 5 s"
                )
                await asyncio.sleep(0.001)

        state = await self.physics_stub.GetSystemState(pb2.Empty())
        assert abs(state.boiler.pressure_pa - 140.0e5) <= 20.0e5
        assert abs(state.boiler.water_level_m - 4.8) <= 0.8

    @pytest.mark.asyncio
    async def test_estop_trips_in_runtime_on_low_water_level(self) -> None:
        params = BoilerParameters()
        low_level_state = replace(params.nominal_initial_state(), water_level=0.4)

        physics_runtime = PhysicsRuntime(
            PhysicsRuntimeConfig(
                dt=1.0,
                initial_state=low_level_state,
                start_paused=True,
            )
        )
        await physics_runtime.start()

        physics_server = grpc.aio.server()
        pb2_grpc.add_PhysicsServiceServicer_to_server(
            PhysicsServicer(physics_runtime),
            physics_server,
        )
        physics_port = physics_server.add_insecure_port("[::]:0")
        await physics_server.start()

        plc_service = PLCService(
            physics_client=PhysicsClient(
                PhysicsClientConfig(target=f"localhost:{physics_port}")
            ),
            control_interval_s=0.05,
            enable_alert_publishing=False,
        )
        await plc_service.start()

        plc_server = grpc.aio.server()
        pb2_grpc.add_PLCServiceServicer_to_server(
            PLCServicer(plc_service),
            plc_server,
        )
        plc_port = plc_server.add_insecure_port("[::]:0")
        await plc_server.start()
        channel = grpc.aio.insecure_channel(f"localhost:{plc_port}")
        stub = pb2_grpc.PLCServiceStub(channel)

        try:
            # Lockstep: one plant step, then the PLC scan of exactly that step, so the
            # outcome does not depend on how fast this machine runs the plant (debt Д2).
            status = await physics_runtime.step(1)
            deadline = time.monotonic() + 5.0
            while plc_service.last_scanned_step < status.step_count:
                assert time.monotonic() < deadline, (
                    f"PLC did not scan step {status.step_count} within 5 s"
                )
                await asyncio.sleep(0.001)
            control_status = await stub.GetControlStatus(pb2.Empty())

            assert control_status.emergency_stop_active
            assert control_status.mode == pb2.ControlMode.ESTOP
            assert control_status.trip_count >= 1
            assert control_status.latest_command.source == pb2.CommandSource.SAFETY
        finally:
            await channel.close()
            await plc_server.stop(grace=0)
            await plc_service.close()
            await physics_server.stop(grace=0)
            await physics_runtime.stop()
