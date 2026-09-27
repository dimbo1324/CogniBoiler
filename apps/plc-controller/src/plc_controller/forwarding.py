"""The path from a PLC decision to the plant: forwarding commands to PhysicsService.

A command counts as in force only once the plant has acknowledged it. Exact repeats of
the command in force are not sent again, and that memory is dropped whenever the plant
may have lost the command: a new plant run, a new state stream, or a plant that reports
valves other than the ones commanded (see `plant_holds`).
"""

from __future__ import annotations

import logging

import grpc.aio

from plc_controller.client import PhysicsClient
from plc_controller.commands import (
    EXTERNAL_COMMAND_SOURCES,
    CommandSnapshot,
    ValidationResult,
)
from plc_controller.measurements import ValveSet
from plc_controller.status import command_msg

logger = logging.getLogger(__name__)

# Commands are compared at the precision they are deduplicated at.
COMMAND_TOLERANCE: float = 1.0e-4
_DEDUP_DIGITS: int = 4

# Failures of the plant link, as opposed to faults of the PLC itself.
LINK_ERRORS: tuple[type[Exception], ...] = (grpc.aio.AioRpcError, ConnectionError)
PLANT_DID_NOT_ACKNOWLEDGE: str = "plant did not acknowledge the command"

type CommandKey = tuple[float, float, float, float, int, str]


def link_error(exc: BaseException) -> str:
    """A one-line reason for a failed call to the plant."""
    if isinstance(exc, grpc.aio.AioRpcError):
        return f"{exc.code().name}: {exc.details()}"
    return str(exc) or type(exc).__name__


def plant_holds(reported: ValveSet, command: CommandSnapshot) -> bool:
    """Does the plant report this command as the one in force? NaN never does."""
    pairs = (
        (reported.fuel, command.fuel_valve),
        (reported.feedwater, command.feedwater_valve),
        (reported.steam, command.steam_valve),
        (reported.spray, command.spray_valve),
    )
    return all(abs(held - sent) <= COMMAND_TOLERANCE for held, sent in pairs)


def _key(snapshot: CommandSnapshot) -> CommandKey:
    return (
        round(snapshot.fuel_valve, _DEDUP_DIGITS),
        round(snapshot.feedwater_valve, _DEDUP_DIGITS),
        round(snapshot.steam_valve, _DEDUP_DIGITS),
        round(snapshot.spray_valve, _DEDUP_DIGITS),
        int(snapshot.source),
        snapshot.operator_id,
    )


class CommandForwarder:
    """Sends commands to PhysicsService and remembers the one in force."""

    def __init__(self, physics: PhysicsClient) -> None:
        self._physics = physics
        self._last_sent: CommandKey | None = None
        self._latest = CommandSnapshot()
        self._failing = False
        self.forwarded = 0
        self.refused = 0
        self.failures = 0

    @property
    def latest(self) -> CommandSnapshot:
        """The command most recently in force (a copy)."""
        return self._latest.copy()

    def invalidate(self) -> None:
        """The plant may have lost the command in force: send the next one anyway."""
        self._last_sent = None

    async def forward(self, snapshot: CommandSnapshot) -> ValidationResult:
        """Send a command, skipping an exact repeat of the one in force."""
        key = _key(snapshot)
        if self._last_sent == key:
            self._latest = snapshot
            return ValidationResult(accepted=True)
        try:
            ack = await self._physics.apply_command(command_msg(snapshot))
        except LINK_ERRORS as exc:
            self._failed(snapshot, exc)
            return ValidationResult(accepted=False, reason=PLANT_DID_NOT_ACKNOWLEDGE)
        if self._failing:
            logger.info("PhysicsService acknowledges commands again")
            self._failing = False
        if not ack.accepted:
            self.refused += 1
            logger.warning("PhysicsService refused a command: %s", ack.reason)
            return ValidationResult(accepted=False, reason=ack.reason)
        self.forwarded += 1
        self._latest = snapshot
        self._last_sent = key
        return ValidationResult(accepted=True)

    def _failed(self, snapshot: CommandSnapshot, exc: Exception) -> None:
        """A person's command is reported every time; the PLC's own once per outage."""
        self.failures += 1
        reason = link_error(exc)
        if int(snapshot.source) in EXTERNAL_COMMAND_SOURCES:
            logger.warning(
                "Command from %s did not reach the plant: %s",
                snapshot.operator_id,
                reason,
            )
        elif not self._failing:
            logger.warning(
                "PhysicsService did not acknowledge the PLC's command: %s — "
                "the next scan sends it again",
                reason,
            )
        self._failing = True
