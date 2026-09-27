"""The PLC's operating modes and their `ControlMode` values in the contract."""

from __future__ import annotations

from enum import StrEnum

import cogniboiler_pb2 as pb2


class RuntimeMode(StrEnum):
    """PLC operating mode."""

    AUTO = "auto"
    MANUAL = "manual"
    ESTOP = "estop"


_TO_PROTO: dict[RuntimeMode, int] = {
    RuntimeMode.AUTO: int(pb2.ControlMode.AUTO),
    RuntimeMode.MANUAL: int(pb2.ControlMode.MANUAL),
    RuntimeMode.ESTOP: int(pb2.ControlMode.ESTOP),
}
_FROM_PROTO: dict[int, RuntimeMode] = {value: mode for mode, value in _TO_PROTO.items()}


def to_proto(mode: RuntimeMode) -> int:
    return _TO_PROTO[mode]


def from_proto(value: int) -> RuntimeMode | None:
    """The mode a contract value names; None for a value the contract does not define."""
    return _FROM_PROTO.get(int(value))
