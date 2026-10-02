"""Invariant I2 in code: the gateway reaches the valves only through the PLC.

The physics stub underneath PhysicsGatewayClient could call any RPC; this test reads the
gateway's source, so a new path to PhysicsService.ApplyControlCommand fails the gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import api_gateway
import pytest

PACKAGE = Path(api_gateway.__file__).resolve().parent
FORBIDDEN = frozenset({"ApplyControlCommand"})


def names_in(source: str) -> set[str]:
    """Attributes, imported names and string constants: every way to name an RPC."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def attribute_names() -> set[str]:
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        names |= names_in(path.read_text(encoding="utf-8"))
    return names


def test_the_gateway_never_calls_the_physics_actuator_rpc() -> None:
    assert attribute_names() & FORBIDDEN == set()


@pytest.mark.parametrize(
    "source",
    [
        "await stub.ApplyControlCommand(command)",
        "call = getattr(stub, 'ApplyControlCommand')",
        "from cogniboiler_pb2_grpc import ApplyControlCommand as apply",
    ],
    ids=["attribute", "getattr", "import"],
)
def test_the_scan_catches_every_way_of_naming_the_actuator_rpc(source: str) -> None:
    assert names_in(source) & FORBIDDEN == {"ApplyControlCommand"}


def test_the_scan_sees_the_rpcs_the_gateway_does_call() -> None:
    assert {
        "SendCommand",
        "ResetEmergencyStop",
        "SetLoadDemand",
        "StreamSystemState",
        "InjectFault",
    } <= attribute_names()
