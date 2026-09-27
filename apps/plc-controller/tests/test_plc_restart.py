"""The restart after a trip on low drum level, reset at the earliest moment it is allowed.

The demo of VISION §7 trips the unit on a failed feedwater pump, repairs the pump and
resets the E-Stop "once the cause has cleared". These tests play that path in lockstep —
one PLC scan for every plant step — and reset on the first step the PLC allows it: the
worst timing an engineer can choose.

In lockstep the restart is clean, but only just: the main steam peaks around 550 °C, 30 K
under the trip. When the PLC's commands reach the plant a step or more late — as they do
with the simulation at ten times real speed — the same restart can trip again on high
main steam temperature. That is recorded as known defect Д17 in the roadmap, with the
evidence; these tests pin the behaviour that holds today so a change to the restart can
be judged against it.
"""

from __future__ import annotations

import asyncio
import logging
import time

import cogniboiler_pb2 as pb2
import pytest
from physics_engine.faults import FaultKind, FaultSpec
from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.service import PLCService, RuntimeMode
from plc_fakes import plc, state
from plc_harness import Rig, rig

MW = 1.0e6
DEMO_LOAD_W = 300.0 * MW
# Steps are one simulated second. The bounds are generous: a slow trip or a slow recovery
# is not what these tests are about.
TRIP_WITHIN_STEPS = 600
RESET_WITHIN_STEPS = 1_200
RECOVERY_STEPS = 900
RECOVERED_W = 150.0 * MW
STEAM_TEMP_TRIP_K = 853.15


async def trip_on_a_failed_feedwater_pump(plant: Rig) -> pb2.PLCStatusMsg:
    ack = await plant.stub.SetLoadDemand(
        pb2.LoadDemandRequest(load_w=DEMO_LOAD_W, operator_id="op")
    )
    assert ack.accepted, ack.reason
    await plant.runtime.inject_fault(FaultSpec(FaultKind.FEEDWATER_PUMP_FAILURE))
    for _ in range(TRIP_WITHIN_STEPS):
        await plant.advance(1)
        status = await plant.stub.GetControlStatus(pb2.Empty())
        if status.emergency_stop_active:
            return status
    raise AssertionError("the failed pump never tripped the unit")


async def reset_as_soon_as_allowed(plant: Rig) -> None:
    """Repair the pump, then reset on the first step the PLC permits it."""
    await plant.runtime.clear_faults()
    for _ in range(RESET_WITHIN_STEPS):
        await plant.advance(1)
        status = await plant.stub.GetControlStatus(pb2.Empty())
        if status.reset_permitted:
            reset = await plant.stub.ResetEmergencyStop(
                pb2.ResetRequest(operator_id="eng")
            )
            assert reset.accepted, reset.reason
            return
    raise AssertionError("the reset was never permitted after the pump was repaired")


class TestRestartAfterALowLevelTrip:
    async def test_the_failed_pump_trips_the_unit_on_the_drum_level(self) -> None:
        async with rig() as plant:
            status = await trip_on_a_failed_feedwater_pump(plant)
            assert status.active_trip.parameter in {"water_level_m", "fuel_permissive"}
            assert status.latest_command.fuel_valve == 0.0

    async def test_a_reset_is_refused_until_the_pump_is_repaired(self) -> None:
        async with rig() as plant:
            await trip_on_a_failed_feedwater_pump(plant)
            for _ in range(60):
                await plant.advance(1)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert not status.reset_permitted
            refused = await plant.stub.ResetEmergencyStop(
                pb2.ResetRequest(operator_id="eng")
            )
            assert not refused.accepted

    async def test_in_lockstep_the_unit_comes_back_on_load_without_a_second_trip(
        self,
    ) -> None:
        async with rig() as plant:
            await trip_on_a_failed_feedwater_pump(plant)
            await reset_as_soon_as_allowed(plant)
            hottest_k = 0.0
            for _ in range(RECOVERY_STEPS):
                await plant.advance(1)
                hottest_k = max(hottest_k, plant.runtime.snapshot.turbine.steam_temp_in)
            status = await plant.stub.GetControlStatus(pb2.Empty())
            assert not status.emergency_stop_active, (
                f"tripped again: {status.active_trip.parameter} "
                f"{status.active_trip.value:.1f} (limit {status.active_trip.threshold:.1f})"
            )
            assert status.trip_count == 1
            assert plant.runtime.snapshot.turbine.electrical_power > RECOVERED_W
            assert hottest_k < STEAM_TEMP_TRIP_K


class TestAPlcRestartWhileTripped:
    """Today's behaviour, pinned so that changing it is a deliberate decision.

    The E-Stop latch lives only in the PLC's memory (audit PLC-03, an owner decision).
    A PLC restarted while the unit is tripped comes back in AUTO with the latch clear
    and no reset, and seeds its load demand from the tripped plant's zero output. The
    fuel stays shut here only because the drum, boxed in by the trip, sits above the
    140 bar pressure setpoint; once the pressure falls below it, AUTO fires again.
    """

    async def test_it_resumes_auto_without_a_reset_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with rig() as plant:
            tripped = await plant.stub.SetControlMode(
                pb2.ControlModeRequest(mode=pb2.ControlMode.ESTOP, operator_id="eng")
            )
            assert tripped.accepted
            await plant.advance(30)
            assert plant.runtime.snapshot.controls.fuel_valve_command == 0.0
            await plant.plc.close()

            restarted = PLCService(
                physics_client=PhysicsClient(
                    PhysicsClientConfig(target=plant.physics_target)
                ),
                retry_delay_s=0.05,
                enable_alert_publishing=False,
            )
            with caplog.at_level(logging.WARNING, logger="plc_controller.service"):
                await restarted.start()
                try:
                    for _ in range(60):
                        await plant.runtime.step(1)
                        await scanned(restarted, plant)
                    status = await restarted.get_control_status()
                finally:
                    await restarted.close()

        assert status.mode == pb2.ControlMode.AUTO
        assert not status.emergency_stop_active
        assert status.trip_count == 0
        assert status.latest_command.source == pb2.CommandSource.PID
        assert status.load_demand_w == 0.0
        assert plant.runtime.snapshot.boiler.pressure > 140.0e5
        assert plant.runtime.snapshot.controls.fuel_valve_command == 0.0
        assert "looks tripped" in caplog.text

    async def test_a_plant_in_operation_raises_no_such_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, _ = plc()
        with caplog.at_level(logging.WARNING, logger="plc_controller.service"):
            await svc.process_state(state(step=0))
            await svc.process_state(state(step=0, run_id=2, fuel_command=0.0))
        assert svc.mode is RuntimeMode.AUTO
        assert "looks tripped" not in caplog.text


async def scanned(svc: PLCService, plant: Rig) -> None:
    target = plant.runtime.simulation_status().step_count
    deadline = time.monotonic() + 10.0
    while svc.last_scanned_step < target:
        assert time.monotonic() < deadline, f"PLC did not scan step {target}"
        await asyncio.sleep(0.001)
