"""Stand-ins for the plant link and the MQTT publisher, for PLC tests without a plant."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import cogniboiler_pb2 as pb2
import grpc
import grpc.aio
from plc_controller.service import PLCService

SENSORS: tuple[str, ...] = (
    "drum_pressure",
    "drum_level",
    "drum_water_temp",
    "furnace_gas_temp",
    "steam_temp",
    "steam_flow",
    "feedwater_flow",
    "fuel_flow",
    "electrical_power",
)

NOMINAL: dict[str, float] = {
    "pressure_pa": 140.0e5,
    "water_level_m": 4.8,
    "water_temp_k": 611.0,
    "flue_gas_temp_k": 1200.0,
    "steam_temp_k": 811.0,
    "steam_flow_kg_s": 245.0,
    "spray_flow_kg_s": 2.0,
    "feedwater_flow_kg_s": 243.0,
    "fuel_flow_kg_s": 17.0,
    "electrical_power_w": 300.0e6,
    "fuel_command": 0.6,
    "feedwater_command": 0.6,
    "steam_command": 0.7,
    "spray_command": 0.05,
}


def state(
    *,
    step: int = 0,
    run_id: int = 1,
    simulation_time_s: float | None = None,
    qualities: dict[str, int] | None = None,
    **overrides: float,
) -> pb2.SystemStateMsg:
    """A plant state at the nominal operating point; step `n` is `n` seconds in."""
    unknown = set(overrides) - set(NOMINAL)
    if unknown:
        raise TypeError(f"unknown state fields: {sorted(unknown)}")
    v = {**NOMINAL, **overrides}
    quality = dict.fromkeys(SENSORS, 0) | (qualities or {})
    return pb2.SystemStateMsg(
        boiler=pb2.BoilerStateMsg(
            pressure_pa=v["pressure_pa"],
            water_level_m=v["water_level_m"],
            water_temp_k=v["water_temp_k"],
            flue_gas_temp_k=v["flue_gas_temp_k"],
            spray_flow_kg_s=v["spray_flow_kg_s"],
            feedwater_flow_kg_s=v["feedwater_flow_kg_s"],
            fuel_flow_kg_s=v["fuel_flow_kg_s"],
        ),
        turbine=pb2.TurbineStateMsg(
            steam_temp_in_k=v["steam_temp_k"],
            steam_flow_kg_s=v["steam_flow_kg_s"],
            electrical_power_w=v["electrical_power_w"],
        ),
        actuators=pb2.ActuatorStateMsg(
            fuel_valve_command=v["fuel_command"],
            feedwater_valve_command=v["feedwater_command"],
            steam_valve_command=v["steam_command"],
            spray_valve_command=v["spray_command"],
        ),
        simulation_time_s=(
            float(step) if simulation_time_s is None else simulation_time_s
        ),
        simulation=pb2.SimulationStatusMsg(step_count=step, run_id=run_id),
        sensors=[
            pb2.SensorStatusMsg(sensor_id=sensor, quality=code)
            for sensor, code in quality.items()
        ],
    )


def rpc_error(
    code: grpc.StatusCode = grpc.StatusCode.UNAVAILABLE,
) -> grpc.aio.AioRpcError:
    """The error a gRPC call raises when the plant cannot be reached."""
    return grpc.aio.AioRpcError(
        code, grpc.aio.Metadata(), grpc.aio.Metadata(), details="plant unreachable"
    )


class FakePhysics:
    """A PhysicsClient that records the commands it is sent; it can fail or refuse."""

    def __init__(self) -> None:
        self.commands: list[pb2.ControlCommandMsg] = []
        self.fail_with: BaseException | None = None
        self.refuse_reason = ""
        self.health_error: BaseException | None = None
        self.feed: asyncio.Queue[pb2.SystemStateMsg | BaseException] = asyncio.Queue()
        self.streams_opened = 0

    async def apply_command(self, command: pb2.ControlCommandMsg) -> pb2.CommandAck:
        if self.fail_with is not None:
            raise self.fail_with
        if self.refuse_reason:
            return pb2.CommandAck(accepted=False, reason=self.refuse_reason)
        self.commands.append(command)
        return pb2.CommandAck(accepted=True)

    async def health(self) -> pb2.HealthStatus:
        if self.health_error is not None:
            raise self.health_error
        return pb2.HealthStatus(service="physics-engine", status="running")

    async def stream_system_state(self) -> AsyncGenerator[pb2.SystemStateMsg]:
        """Yields the states fed to it; a fed exception breaks the stream."""
        self.streams_opened += 1
        while True:
            item = await self.feed.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self) -> None:
        return None


class RecordingPublisher:
    """A PlcPublisher that keeps what it would have published."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.alarms: list[Any] = []
        self.dropped = 0
        self.connected = False

    def publish_event(self, event: Any) -> None:
        self.events.append(event)

    def publish_alarm(self, transition: Any) -> None:
        self.alarms.append(transition)

    def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def plc(physics: FakePhysics | None = None) -> tuple[PLCService, FakePhysics]:
    """A PLC without a scan loop or a broker, wired to a fake plant."""
    fake = physics or FakePhysics()
    service = PLCService(
        physics_client=fake,  # type: ignore[arg-type]
        enable_control_loop=False,
        enable_alert_publishing=False,
    )
    return service, fake
