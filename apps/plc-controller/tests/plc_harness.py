"""The PLC on a paused plant, advanced in lockstep: one plant step, then its PLC scan."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from physics_engine.models import BoilerState
from physics_engine.runtime import PhysicsRuntime, PhysicsRuntimeConfig
from physics_engine.scenarios import ScenarioName
from physics_engine.server import PhysicsServicer
from plc_controller.client import PhysicsClient, PhysicsClientConfig
from plc_controller.server import PLCServicer
from plc_controller.service import PLCService


@dataclass
class Rig:
    runtime: PhysicsRuntime
    plc: PLCService
    stub: pb2_grpc.PLCServiceStub
    physics_target: str

    async def settle(self) -> None:
        """Wait until the PLC has scanned the plant's latest step."""
        target = self.runtime.simulation_status().step_count
        deadline = time.monotonic() + 10.0
        while self.plc.last_scanned_step < target:
            assert time.monotonic() < deadline, f"PLC did not scan step {target}"
            await asyncio.sleep(0.001)

    async def advance(self, steps: int) -> None:
        """Step the plant one step at a time, each followed by its PLC scan."""
        for _ in range(steps):
            await self.runtime.step(1)
            await self.settle()


@asynccontextmanager
async def rig(
    scenario: ScenarioName = ScenarioName.STEADY_STATE,
    initial_state: BoilerState | None = None,
) -> AsyncIterator[Rig]:
    runtime = PhysicsRuntime(
        PhysicsRuntimeConfig(
            scenario=scenario, dt=1.0, start_paused=True, initial_state=initial_state
        )
    )
    await runtime.start()
    physics_server = grpc.aio.server()
    pb2_grpc.add_PhysicsServiceServicer_to_server(
        PhysicsServicer(runtime), physics_server
    )
    physics_port = physics_server.add_insecure_port("127.0.0.1:0")
    await physics_server.start()
    target = f"127.0.0.1:{physics_port}"
    plc = PLCService(
        physics_client=PhysicsClient(PhysicsClientConfig(target=target)),
        control_interval_s=0.05,
        enable_alert_publishing=False,
    )
    await plc.start()
    plc_server = grpc.aio.server()
    pb2_grpc.add_PLCServiceServicer_to_server(PLCServicer(plc), plc_server)
    plc_port = plc_server.add_insecure_port("127.0.0.1:0")
    await plc_server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{plc_port}")
    found = Rig(runtime, plc, pb2_grpc.PLCServiceStub(channel), target)
    try:
        await found.settle()
        yield found
    finally:
        await channel.close()
        await plc_server.stop(grace=None)
        await plc.close()
        await physics_server.stop(grace=None)
        await runtime.stop()
