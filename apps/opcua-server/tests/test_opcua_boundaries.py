"""The OPC UA server reads PLCService and AlarmService; it never commands them (I11).

Every write from OPC UA goes through the gateway as the signed-in user. The read-only
clients hold full gRPC stubs, so the boundary is checked in the source: no command RPC
and no PhysicsService stub may be named anywhere in the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import opcua_server
import pytest

PACKAGE = Path(opcua_server.__file__).parent

FORBIDDEN = frozenset(
    {
        "SendCommand",
        "UpdateSetpoints",
        "ResetEmergencyStop",
        "SetLoadDemand",
        "SetControlMode",
        "AcknowledgeAlarm",
        "AcknowledgeAll",
        "ApplyControlCommand",
        "PhysicsServiceStub",
    }
)
EXPECTED_READS = frozenset({"GetControlStatus", "ListAlarms"})


def names_in(source: str) -> set[str]:
    """Every attribute accessed, name loaded and name imported in `source`.

    String constants are not scanned: the OPC UA methods carry the same names as the
    PLC and alarm RPCs they forward to the gateway.
    """
    found: set[str] = set()
    for item in ast.walk(ast.parse(source)):
        if isinstance(item, ast.Attribute):
            found.add(item.attr)
        elif isinstance(item, ast.Name):
            found.add(item.id)
        elif isinstance(item, ast.alias):
            found.add(item.name.rsplit(".", 1)[-1])
    return found


def names_used() -> set[str]:
    """Every attribute accessed and every name loaded in the package's code."""
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        found |= names_in(path.read_text(encoding="utf-8"))
    return found


def test_the_package_names_no_command_rpc() -> None:
    assert names_used() & FORBIDDEN == set()


@pytest.mark.parametrize(
    ("source", "named"),
    [
        ("await self._stub.SendCommand(request)", "SendCommand"),
        ("stub = pb2_grpc.PhysicsServiceStub(channel)", "PhysicsServiceStub"),
        (
            "from cogniboiler_pb2_grpc import PhysicsServiceStub as Physics",
            "PhysicsServiceStub",
        ),
        ("import cogniboiler_pb2_grpc.AcknowledgeAll", "AcknowledgeAll"),
    ],
    ids=["rpc-call", "stub", "imported-stub", "dotted-import"],
)
def test_the_scan_catches_a_command_path(source: str, named: str) -> None:
    assert names_in(source) & FORBIDDEN == {named}


def test_the_scan_sees_the_reads_the_server_makes() -> None:
    assert EXPECTED_READS <= names_used()
