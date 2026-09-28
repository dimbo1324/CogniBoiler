"""
Async gRPC client for the PhysicsService.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import cogniboiler_pb2 as pb2
import cogniboiler_pb2_grpc as pb2_grpc
import grpc.aio
from cogniboiler_observability import observed_channel

DEFAULT_PHYSICS_TARGET: str = "localhost:50052"

# The gateway gives a PLCService call 2 s. An E-Stop can wait behind a scan's command
# to the plant and then send its own, so each plant call must take under half of it.
DEFAULT_PHYSICS_TIMEOUT_S: float = 0.75

# HTTP/2 keepalive finds a plant that vanished without closing the connection, which
# would otherwise leave the state stream waiting forever. A paused plant sends no data;
# PhysicsService accepts pings every 5 s with or without data, so the PLC pings every
# 10 s and gives up on a plant that has not answered within 5 s. Without
# max_pings_without_data=0, grpc-core stops pinging after two pings on an idle stream.
KEEPALIVE_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
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
            self._channel = observed_channel(
                self.config.target, options=KEEPALIVE_OPTIONS
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
