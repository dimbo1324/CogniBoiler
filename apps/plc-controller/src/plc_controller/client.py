"""
Async gRPC client for the PhysicsService.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio

DEFAULT_PHYSICS_TARGET: str = "localhost:50052"


@dataclass
class PhysicsClientConfig:
    """Connection parameters for the PhysicsService."""

    target: str = DEFAULT_PHYSICS_TARGET
    timeout_s: float = 2.0


class PhysicsClient:
    """Small async wrapper around the generated PhysicsServiceStub."""

    def __init__(self, config: PhysicsClientConfig | None = None) -> None:
        self.config = config or PhysicsClientConfig()
        self._channel: grpc.aio.Channel | None = None
        self._stub: pb2_grpc.PhysicsServiceStub | None = None

    def _connected_stub(self) -> pb2_grpc.PhysicsServiceStub:
        """The stub, creating the gRPC channel lazily."""
        if self._channel is None or self._stub is None:
            self._channel = grpc.aio.insecure_channel(self.config.target)
            self._stub = pb2_grpc.PhysicsServiceStub(self._channel)
        return self._stub

    async def connect(self) -> None:
        """Create the gRPC channel lazily."""
        self._connected_stub()

    async def close(self) -> None:
        """Close the gRPC channel if it was opened."""
        if self._channel is None:
            return
        await self._channel.close()
        self._channel = None
        self._stub = None

    async def health(self) -> pb2.HealthStatus:
        """Fetch PhysicsService health."""
        return await self._connected_stub().Health(
            pb2.Empty(), timeout=self.config.timeout_s
        )

    async def get_system_state(self) -> pb2.SystemStateMsg:
        """Fetch the current live process state."""
        return await self._connected_stub().GetSystemState(
            pb2.Empty(),
            timeout=self.config.timeout_s,
        )

    async def stream_system_state(self) -> AsyncIterator[pb2.SystemStateMsg]:
        """Every plant state the PhysicsService publishes, as it is published."""
        call = self._connected_stub().StreamSystemState(
            pb2.StreamRequest(interval_s=0.0)
        )
        try:
            async for state in call:
                yield state
        finally:
            call.cancel()

    async def apply_command(self, command: pb2.ControlCommandMsg) -> pb2.CommandAck:
        """Forward a validated control command to the PhysicsService."""
        return await self._connected_stub().ApplyControlCommand(
            command,
            timeout=self.config.timeout_s,
        )
