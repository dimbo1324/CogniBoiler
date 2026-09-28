"""The OPC UA server reads PLCService and AlarmService; it never commands them (I11).

Every write from OPC UA goes through the gateway as the signed-in user. The read-only
clients hold full gRPC stubs, so the boundary is checked in the source: no command RPC
and no PhysicsService stub may be named anywhere in the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import opcua_server

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


def names_used() -> set[str]:
    """Every attribute accessed and every name loaded in the package's code."""
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for item in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(item, ast.Attribute):
                found.add(item.attr)
            elif isinstance(item, ast.Name):
                found.add(item.id)
    return found


def test_the_package_names_no_command_rpc() -> None:
    assert names_used() & FORBIDDEN == set()


def test_the_scan_sees_the_reads_the_server_makes() -> None:
    assert EXPECTED_READS <= names_used()
