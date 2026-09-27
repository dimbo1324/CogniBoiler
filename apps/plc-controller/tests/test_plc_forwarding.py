"""The path from a PLC decision to the plant, against a fake plant link.

A command is only in force once the plant has it. These tests cover what happens when
the plant loses it behind the PLC's back (a scenario load, a physics restart), when the
link fails under a command, and when the scan loop itself breaks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import cogniboiler_pb2 as pb2
import grpc
import pytest
from plc_controller.client import PhysicsClientConfig
from plc_controller.service import PLCService, RuntimeMode
from plc_fakes import FakePhysics, plc, rpc_error, state

# The gateway's deadline for a PLCService call (api_gateway.clients).
GATEWAY_PLC_DEADLINE_S = 2.0

OVERFLOW = 7.9
SHUT: dict[str, float] = {
    "fuel_command": 0.0,
    "feedwater_command": 0.0,
    "steam_command": 0.0,
    "spray_command": 0.0,
}


async def trip_on_high_level(svc: PLCService, physics: FakePhysics) -> None:
    await svc.process_state(state(step=0, water_level_m=OVERFLOW))
    assert svc.mode is RuntimeMode.ESTOP
    last = physics.commands[-1]
    assert (last.fuel_valve, last.feedwater_valve, last.steam_valve) == (0.0, 0.0, 0.0)


@asynccontextmanager
async def scanning(physics: FakePhysics) -> AsyncIterator[PLCService]:
    """A PLC whose scan loop reads the fake plant's stream."""
    svc = PLCService(
        physics_client=physics,  # type: ignore[arg-type]
        control_interval_s=0.01,
        enable_alert_publishing=False,
    )
    await svc.start()
    try:
        yield svc
    finally:
        await svc.close()


async def scans(svc: PLCService, count: int) -> None:
    async with asyncio.timeout(5.0):
        while svc.stats["scans"] < count:
            await asyncio.sleep(0.001)


class TestLatchedTripIsReSent:
    async def test_a_new_plant_run_gets_the_trip_command_again(self) -> None:
        svc, physics = plc()
        await trip_on_high_level(svc, physics)
        sent = len(physics.commands)
        # A scenario load resets the plant's valves to the scenario's own.
        await svc.process_state(
            state(step=0, run_id=2, water_level_m=OVERFLOW, fuel_command=0.6)
        )
        assert len(physics.commands) == sent + 1
        assert physics.commands[-1].fuel_valve == 0.0
        assert physics.commands[-1].source == pb2.CommandSource.SAFETY

    async def test_a_plant_that_lost_the_trip_command_gets_it_again(self) -> None:
        svc, physics = plc()
        await trip_on_high_level(svc, physics)
        await svc.process_state(state(step=1, water_level_m=OVERFLOW, **SHUT))
        sent = len(physics.commands)
        # Same run, but the plant reports a fuel valve the trip never commanded.
        await svc.process_state(
            state(step=2, water_level_m=OVERFLOW, **(SHUT | {"fuel_command": 0.6}))
        )
        assert len(physics.commands) == sent + 1
        assert physics.commands[-1].fuel_valve == 0.0

    async def test_a_plant_that_holds_the_trip_command_is_not_sent_it_again(
        self,
    ) -> None:
        svc, physics = plc()
        await trip_on_high_level(svc, physics)
        for step in (1, 2, 3):
            await svc.process_state(state(step=step, water_level_m=OVERFLOW, **SHUT))
        assert len(physics.commands) == 1

    async def test_a_reconnected_stream_gets_the_trip_command_again(self) -> None:
        physics = FakePhysics()
        async with scanning(physics) as svc:
            await physics.feed.put(state(step=0, water_level_m=OVERFLOW))
            await scans(svc, 1)
            assert len(physics.commands) == 1
            # The physics process restarts: the stream breaks, the run id is 1 again
            # and the plant reports shut valves, so only the reconnect says "resend".
            await physics.feed.put(ConnectionError("physics restarted"))
            await physics.feed.put(state(step=0, water_level_m=OVERFLOW, **SHUT))
            await scans(svc, 2)
            assert physics.streams_opened == 2
            assert len(physics.commands) == 2
            assert physics.commands[-1].fuel_valve == 0.0


async def operator_command(svc: PLCService, fuel: float = 0.4) -> object:
    return await svc.send_command(
        fuel_valve=fuel,
        feedwater_valve=0.5,
        steam_valve=0.6,
        source=pb2.CommandSource.OPERATOR,
        operator_id="anna",
    )


class TestCommandsThePlantNeverGot:
    async def test_an_unreachable_plant_leaves_the_plc_in_auto(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        physics.fail_with = rpc_error()
        with caplog.at_level(logging.WARNING, logger="plc_controller"):
            result = await operator_command(svc)
        assert not result.accepted
        assert "plant did not acknowledge the command" in result.reason
        assert svc.mode is RuntimeMode.AUTO
        assert svc.stats["commands_forwarded"] == 0
        assert svc.latest_command().operator_id != "anna"
        assert "anna" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_a_command_the_plant_refuses_leaves_the_plc_in_auto(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        physics.refuse_reason = "valve range"
        with caplog.at_level(logging.WARNING, logger="plc_controller"):
            result = await operator_command(svc)
        assert (result.accepted, result.reason) == (False, "valve range")
        assert svc.mode is RuntimeMode.AUTO
        assert svc.stats["commands_forwarded"] == 0
        assert "PhysicsService refused a command: valve range" in caplog.text

    async def test_a_delivered_command_switches_to_manual(self) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        result = await operator_command(svc)
        assert result.accepted
        assert svc.mode is RuntimeMode.MANUAL
        assert physics.commands[-1].operator_id == "anna"


class TestEmergencyStopThePlantNeverGot:
    async def test_the_latch_holds_and_the_caller_is_told_so(self) -> None:
        svc, physics = plc()
        await svc.process_state(state(step=0))
        physics.fail_with = rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
        result = await svc.set_mode(RuntimeMode.ESTOP, "eng")
        assert result.accepted
        assert "E-Stop latched" in result.reason
        assert svc.mode is RuntimeMode.ESTOP
        # The next scan, with the plant back, sends the trip command.
        physics.fail_with = None
        await svc.process_state(state(step=1))
        assert physics.commands[-1].source == pb2.CommandSource.SAFETY
        assert physics.commands[-1].fuel_valve == 0.0

    def test_two_plant_calls_answer_inside_the_gateway_deadline(self) -> None:
        # An E-Stop can wait for a scan's command to the plant (the PLC's lock) and
        # then send its own: both must fit in one gateway call.
        assert 2 * PhysicsClientConfig().timeout_s < GATEWAY_PLC_DEADLINE_S
