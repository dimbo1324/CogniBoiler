"""gRPC clients of the upstream services: physics, PLC and alarms."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from cogniboiler_observability import client_interceptors


@dataclass
class PhysicsGatewayConfig:
    """Connection settings for the live PhysicsService."""

    target: str = "localhost:50052"
    timeout_s: float = 2.0


@dataclass
class PLCGatewayConfig:
    """Connection settings for the live PLCService."""

    target: str = "localhost:50051"
    timeout_s: float = 2.0


@dataclass
class AlarmGatewayConfig:
    """Connection settings for the alert-manager AlarmService."""

    target: str = "localhost:50053"
    timeout_s: float = 3.0


# The telemetry stream has no deadline, and a paused plant sends nothing on it: without
# pings a frozen engine or a silently dropped link would stall it for good. gRPC servers
# refuse pings more often than every 5 minutes unless told otherwise, so 6 minutes; any
# number of pings may go out while the stream is quiet.
PHYSICS_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 360_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.http2.max_pings_without_data", 0),
)


class PhysicsGatewayClient:
    """
    Async wrapper around the generated PhysicsServiceStub.

    It reads the plant and controls the simulation. It never sends valve commands:
    those go through the PLC.
    """

    def __init__(self, config: PhysicsGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target,
            options=PHYSICS_CHANNEL_OPTIONS,
            interceptors=client_interceptors(),
        )
        self._stub = pb2_grpc.PhysicsServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def get_system_state(self) -> pb2.SystemStateMsg:
        return await self._stub.GetSystemState(
            pb2.Empty(),
            timeout=self.config.timeout_s,
        )

    async def stream_system_state(
        self,
        *,
        interval_s: float = 0.0,
    ) -> AsyncGenerator[pb2.SystemStateMsg]:
        stream = self._stub.StreamSystemState(
            pb2.StreamRequest(interval_s=interval_s),
            timeout=None,
        )
        async for item in stream:
            yield item

    async def get_simulation_status(self) -> pb2.SimulationStatusMsg:
        return await self._stub.GetSimulationStatus(
            pb2.Empty(), timeout=self.config.timeout_s
        )

    async def pause(self, operator_id: str) -> pb2.SimulationAck:
        return await self._stub.PauseSimulation(
            pb2.SimulationControlRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def resume(self, operator_id: str) -> pb2.SimulationAck:
        return await self._stub.ResumeSimulation(
            pb2.SimulationControlRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_speed(
        self, speed_factor: float, operator_id: str
    ) -> pb2.SimulationAck:
        return await self._stub.SetSimulationSpeed(
            pb2.SimulationSpeedRequest(
                speed_factor=speed_factor, operator_id=operator_id
            ),
            timeout=self.config.timeout_s,
        )

    async def step(self, steps: int, operator_id: str) -> pb2.SimulationAck:
        # Stepping publishes every step and may take a while for long requests.
        return await self._stub.StepSimulation(
            pb2.StepRequest(steps=steps, operator_id=operator_id),
            timeout=max(self.config.timeout_s, 60.0),
        )

    async def list_scenarios(self) -> pb2.ScenarioListMsg:
        return await self._stub.ListScenarios(
            pb2.Empty(), timeout=self.config.timeout_s
        )

    async def load_scenario(self, name: str, operator_id: str) -> pb2.SimulationAck:
        # Loading solves an operating point before it answers.
        return await self._stub.LoadScenario(
            pb2.ScenarioRequest(name=name, operator_id=operator_id),
            timeout=max(self.config.timeout_s, 30.0),
        )

    async def inject_fault(self, request: pb2.FaultRequest) -> pb2.FaultAck:
        return await self._stub.InjectFault(request, timeout=self.config.timeout_s)

    async def clear_fault(self, request: pb2.FaultClearRequest) -> pb2.FaultAck:
        return await self._stub.ClearFault(request, timeout=self.config.timeout_s)


class PLCGatewayClient:
    """Small async wrapper around the generated PLCServiceStub."""

    def __init__(self, config: PLCGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target, interceptors=client_interceptors()
        )
        self._stub = pb2_grpc.PLCServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def send_command(
        self,
        command: pb2.ControlCommandMsg,
    ) -> pb2.CommandAck:
        return await self._stub.SendCommand(command, timeout=self.config.timeout_s)

    async def update_setpoints(self, setpoints: pb2.SetpointsMsg) -> pb2.CommandAck:
        return await self._stub.UpdateSetpoints(
            setpoints, timeout=self.config.timeout_s
        )

    async def get_control_status(self) -> pb2.PLCStatusMsg:
        return await self._stub.GetControlStatus(
            pb2.Empty(),
            timeout=self.config.timeout_s,
        )

    async def reset_emergency_stop(self, operator_id: str) -> pb2.CommandAck:
        return await self._stub.ResetEmergencyStop(
            pb2.ResetRequest(operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_load_demand(self, load_w: float, operator_id: str) -> pb2.CommandAck:
        return await self._stub.SetLoadDemand(
            pb2.LoadDemandRequest(load_w=load_w, operator_id=operator_id),
            timeout=self.config.timeout_s,
        )

    async def set_control_mode(self, mode: int, operator_id: str) -> pb2.CommandAck:
        return await self._stub.SetControlMode(
            pb2.ControlModeRequest(mode=mode, operator_id=operator_id),
            timeout=self.config.timeout_s,
        )


class AlarmGatewayClient:
    """Small async wrapper around the generated AlarmServiceStub."""

    def __init__(self, config: AlarmGatewayConfig) -> None:
        self.config = config
        self._channel = grpc.aio.insecure_channel(
            config.target, interceptors=client_interceptors()
        )
        self._stub = pb2_grpc.AlarmServiceStub(self._channel)

    async def close(self) -> None:
        await self._channel.close()

    async def health(self) -> pb2.HealthStatus:
        return await self._stub.Health(pb2.Empty(), timeout=self.config.timeout_s)

    async def list_alarms(self, request: pb2.ListAlarmsRequest) -> pb2.AlarmListMsg:
        return await self._stub.ListAlarms(request, timeout=self.config.timeout_s)

    async def get_alarm(self, alarm_id: int) -> pb2.AlarmDetailMsg:
        return await self._stub.GetAlarm(
            pb2.AlarmRef(alarm_id=alarm_id), timeout=self.config.timeout_s
        )

    async def acknowledge(
        self, alarm_id: int, operator_id: str, comment: str
    ) -> pb2.AcknowledgeResult:
        return await self._stub.AcknowledgeAlarm(
            pb2.AcknowledgeAlarmRequest(
                alarm_id=alarm_id, operator_id=operator_id, comment=comment
            ),
            timeout=self.config.timeout_s,
        )

    async def acknowledge_all(
        self, operator_id: str, comment: str, severity: str
    ) -> pb2.AcknowledgeResult:
        return await self._stub.AcknowledgeAll(
            pb2.AcknowledgeAllRequest(
                operator_id=operator_id, comment=comment, severity=severity
            ),
            timeout=self.config.timeout_s,
        )
