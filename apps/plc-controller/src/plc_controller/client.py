"""
Async gRPC client for the PhysicsService.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from cogniboiler_observability import client_interceptors

DEFAULT_PHYSICS_TARGET: str = "localhost:50052"

# The gateway gives a PLCService call 2 s. An E-Stop can wait behind a scan's command
# to the plant and then send its own, so each plant call must take under half of it.
DEFAULT_PHYSICS_TIMEOUT_S: float = 0.75

# HTTP/2 keepalive finds a plant that vanished without closing the connection, which
# would otherwise leave the state stream waiting forever. A paused plant sends no data,
# and a gRPC server with default options answers pings on such an idle stream more often
# than every 5 minutes with GOAWAY "too_many_pings", so the interval stays above that.
KEEPALIVE_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 360_000),
    ("grpc.keepalive_timeout_ms", 20_000),
    ("grpc.keepalive_permit_without_calls", 0),
)


@dataclass
class PhysicsClientConfig:
    """Connection parameters for the PhysicsService."""

    target: str = DEFAULT_PHYSICS_TARGET
    timeout_s: float = DEFAULT_PHYSICS_TIMEOUT_S


class PhysicsClient:
    """Small async wrapper around the generated PhysicsServiceStub."""

    def __init__(self, config: PhysicsClientConfig | None = None) -> None:
        self.config = config or PhysicsClientConfig()
        self._channel: grpc.aio.Channel | None = None
        self._stub: pb2_grpc.PhysicsServiceStub | None = None

    def _connected_stub(self) -> pb2_grpc.PhysicsServiceStub:
        """The stub, creating the gRPC channel lazily."""
        if self._channel is None or self._stub is None:
            self._channel = grpc.aio.insecure_channel(
                self.config.target,
                options=list(KEEPALIVE_OPTIONS),
                interceptors=client_interceptors(),
            )
            self._stub = pb2_grpc.PhysicsServiceStub(self._channel)
        return self._stub

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

    async def stream_system_state(self) -> AsyncGenerator[pb2.SystemStateMsg]:
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
