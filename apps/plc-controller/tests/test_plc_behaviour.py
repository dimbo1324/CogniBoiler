"""The PLC against the live plant, in lockstep: load, modes, trips, resets, runs, RPCs."""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import replace

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
import pytest
from aiomqtt import MqttError
from physics_engine.models import BoilerParameters
from physics_engine.scenarios import ScenarioName
from plc_controller import events as plc_events
from plc_controller import server as plc_server
from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.metrics import PlcCollector
from plc_controller.service import PLCService, RuntimeMode
from plc_harness import rig

MW = 1.0e6


class TestLoadFollowing:
    async def test_the_unit_follows_a_load_demand_in_auto(self) -> None:
        async with rig() as plant:
            start = plant.runtime.snapshot.turbine.electrical_power
            assert plant.plc.load_demand_w == pytest.approx(start)
            ack = await plant.stub.SetLoadDemand(
                pb2.LoadDemandRequest(load_w=200 * MW, operator_id="operator1")
            )
            assert ack.accepted
            await plant.advance(60)
            ramping = await plant.stub.GetControlStatus(pb2.Empty())
            assert ramping.load_demand_w == 200 * MW
            assert 200 * MW < ramping.load_setpoint_w < start
            await plant.advance(420)
            power = plant.runtime.snapshot.turbine.electrical_power
            assert power == pytest.approx(200 * MW, abs=6 * MW)

    @pytest.mark.parametrize("load_w", [-1.0, 350 * MW])
    async def test_a_load_outside_the_rating_is_refused(self, load_w: float) -> None:
        async with rig() as plant:
            ack = await plant.stub.SetLoadDemand(pb2.LoadDemandRequest(load_w=load_w))
            assert not ack.accepted
            assert "Load demand" in ack.reason

    async def test_the_status_shows_the_loops_and_working_setpoints(self) -> None:
        async with rig() as plant:
            await plant.advance(3)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.mode == pb2.ControlMode.AUTO
            assert {loop.name for loop in status.loops} >= {
                "load",
                "pressure",
                "drum_level",
            }
            assert status.active_setpoints.pressure_pa > 0
            assert status.latest_command.source == pb2.CommandSource.PID
            assert status.run_id == plant.runtime.simulation_status().run_id


class TestManualMode:
    async def test_an_operator_command_switches_to_manual_and_holds(self) -> None:
        async with rig() as plant:
            ack = await plant.stub.SendCommand(
                pb2.ControlCommandMsg(
                    fuel_valve=0.4,
                    feedwater_valve=0.5,
                    steam_valve=0.6,
                    source=pb2.CommandSource.OPERATOR,
                    operator_id="operator1",
                )
            )
            assert ack.accepted
            await plant.advance(10)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.mode == pb2.ControlMode.MANUAL
            assert status.latest_command.source == pb2.CommandSource.OPERATOR
            assert status.latest_command.fuel_valve == pytest.approx(0.4)

            back = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.AUTO, operator_id="op")
            )
            assert back.accepted
            await plant.advance(3)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.mode == pb2.ControlMode.AUTO
            assert status.latest_command.source == pb2.CommandSource.PID

    async def test_switching_to_manual_keeps_the_last_command(self) -> None:
        async with rig() as plant:
            await plant.advance(2)
            before = plant.plc.latest_command()
            ack = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.MANUAL)
            )
            assert ack.accepted
            await plant.advance(5)
            assert plant.plc.mode is RuntimeMode.MANUAL
            assert plant.plc.latest_command().fuel_valve == pytest.approx(
                before.fuel_valve
            )

    @pytest.mark.parametrize(
        "source", [pb2.CommandSource.PID, pb2.CommandSource.SAFETY]
    )
    async def test_sources_reserved_for_the_plc_are_refused(self, source: int) -> None:
        async with rig() as plant:
            ack = await plant.stub.SendCommand(
                pb2.ControlCommandMsg(
                    fuel_valve=0.4, feedwater_valve=0.5, steam_valve=0.6, source=source
                )
            )
            assert not ack.accepted
            assert "reserved for the PLC" in ack.reason
            assert plant.plc.stats["commands_rejected"] == 1

    async def test_an_unknown_mode_is_refused(self) -> None:
        async with rig() as plant:
            ack = await plant.stub.SetControlMode(pb2.ControlModeRequest(mode=9))
            assert (ack.accepted, ack.reason) == (False, "unknown control mode 9")


class TestTripsAndResets:
    async def test_a_manual_trip_latches_until_reset(self) -> None:
        async with rig() as plant:
            tripped = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.ESTOP, operator_id="eng")
            )
            assert tripped.accepted
            await plant.advance(2)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.emergency_stop_active
            assert status.active_trip.parameter == "manual_trip"
            assert status.latest_command.source == pb2.CommandSource.SAFETY
            assert status.latest_command.fuel_valve == 0.0

            refused = await plant.stub.SendCommand(
                pb2.ControlCommandMsg(
                    fuel_valve=0.5, feedwater_valve=0.5, steam_valve=0.5
                )
            )
            assert not refused.accepted and "Emergency stop is active" in refused.reason
            auto = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.AUTO)
            )
            assert not auto.accepted and "Reset it first" in auto.reason
            again = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.ESTOP)
            )
            assert again.accepted
            assert plant.plc.stats["trips"] == 1

            health = await plant.stub.Health(pb2.Empty())
            assert health.status == "degraded"

            reset = await plant.stub.ResetEmergencyStop(
                pb2.ResetRequest(operator_id="eng")
            )
            assert reset.accepted, reset.reason
            await plant.advance(2)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert (status.mode, status.emergency_stop_active) == (
                pb2.ControlMode.AUTO,
                False,
            )
            assert status.active_trip.parameter == ""

    async def test_resetting_without_a_trip_is_a_no_op(self) -> None:
        async with rig() as plant:
            ack = await plant.stub.ResetEmergencyStop(pb2.ResetRequest())
            assert (ack.accepted, ack.reason) == (True, "Emergency stop is not active.")

    async def test_a_reset_is_refused_while_the_cause_persists(self) -> None:
        low = replace(BoilerParameters().nominal_initial_state(), water_level=0.4)
        async with rig(initial_state=low) as plant:
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.emergency_stop_active
            assert not status.reset_permitted
            assert status.reset_blockers
            ack = await plant.stub.ResetEmergencyStop(
                pb2.ResetRequest(operator_id="eng")
            )
            assert not ack.accepted
            assert ack.reason.startswith("Reset refused: ")
            assert plant.plc.mode is RuntimeMode.ESTOP

    async def test_before_any_plant_state_a_reset_is_blocked(self) -> None:
        plc = PLCService(
            physics_client=PhysicsClient(PhysicsClientConfig(target="127.0.0.1:1")),
            enable_control_loop=False,
            enable_alert_publishing=False,
        )
        try:
            await plc.set_mode(RuntimeMode.ESTOP, "eng")
            result = await plc.reset_emergency_stop("eng")
            assert not result.accepted
            assert "no plant state received yet" in result.reason
        finally:
            await plc.close()


class TestRuns:
    async def test_a_new_plant_run_reseeds_the_load_demand(self) -> None:
        async with rig() as plant:
            await plant.stub.SetLoadDemand(pb2.LoadDemandRequest(load_w=200 * MW))
            await plant.runtime.load_scenario(ScenarioName.PART_LOAD)
            # Step counts restart with the run; one scan past the load reaches it.
            await plant.advance(1)
            assert plant.plc.load_demand_w == pytest.approx(
                plant.runtime.snapshot.turbine.electrical_power
            )
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert status.run_id == plant.runtime.simulation_status().run_id


class TestConnection:
    async def test_the_scan_waits_for_the_plant_and_recovers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        plc = PLCService(
            physics_client=PhysicsClient(
                PhysicsClientConfig(target=f"127.0.0.1:{port}", timeout_s=0.5)
            ),
            control_interval_s=0.05,
            enable_alert_publishing=False,
        )
        with caplog.at_level(logging.INFO, logger="plc_controller.service"):
            await plc.start()
            try:
                async with asyncio.timeout(10.0):
                    while "PLC scan stream failed" not in caplog.text:
                        await asyncio.sleep(0.01)
                assert await plc.physics_status() == "degraded"
            finally:
                await plc.close()
        assert caplog.text.count("PLC scan stream failed") == 1


class TestMetrics:
    async def test_the_collector_reports_counters_mode_and_conditions(self) -> None:
        async with rig() as plant:
            await plant.advance(2)
            metrics = {
                sample.name: sample.value
                for family in PlcCollector(plant.plc).collect()
                for sample in family.samples
                if not sample.labels or sample.labels.get("mode") == "auto"
            }
            assert metrics["plc_scans_total"] >= 2
            assert metrics["plc_mode"] == 1.0
            assert metrics["plc_alarm_conditions_active"] == 0.0


class TestCommandStream:
    async def test_the_latest_command_is_streamed(self) -> None:
        async with rig() as plant:
            await plant.advance(1)
            stream = plant.stub.StreamCommands(pb2.StreamRequest(interval_s=0.1))
            first = await stream.read()
            second = await stream.read()
            stream.cancel()
            assert first.source == second.source == pb2.CommandSource.PID
            assert first.operator_id == "plc-auto"


class TestServe:
    async def test_serve_answers_and_closes_the_service_when_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[bool] = []
        original_close = PLCService.close

        async def close(self: PLCService) -> None:
            closed.append(True)
            await original_close(self)

        class Broker:
            def __init__(self, **_: object) -> None:
                return None

            async def __aenter__(self) -> Broker:
                raise MqttError("no broker in this test")

            async def __aexit__(self, *_: object) -> None:
                return None

        monkeypatch.setattr(PLCService, "close", close)
        monkeypatch.setattr(plc_server, "observe_service", lambda service: None)
        monkeypatch.setattr(plc_events, "Client", Broker)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        async with rig() as plant:
            task = asyncio.create_task(
                plc_server.serve(port, physics_target=plant.physics_target)
            )
            try:
                async with asyncio.timeout(10.0):
                    while True:
                        try:
                            async with grpc.aio.insecure_channel(
                                f"127.0.0.1:{port}"
                            ) as channel:
                                health = await pb2_grpc.PLCServiceStub(channel).Health(
                                    pb2.Empty(), timeout=1.0
                                )
                            break
                        except grpc.aio.AioRpcError:
                            await asyncio.sleep(0.05)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert closed == [True]
        assert health.service == "plc-controller"
