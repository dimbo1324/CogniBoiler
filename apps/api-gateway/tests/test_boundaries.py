"""Invariant I2 in code: the gateway reaches the valves only through the PLC.

The physics stub underneath PhysicsGatewayClient could call any RPC; this test reads the
gateway's source, so a new path to PhysicsService.ApplyControlCommand fails the gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import api_gateway

PACKAGE = Path(api_gateway.__file__).resolve().parent
FORBIDDEN = frozenset({"ApplyControlCommand"})


def attribute_names() -> set[str]:
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )
        names.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    return names


def test_the_gateway_never_calls_the_physics_actuator_rpc() -> None:
    assert attribute_names() & FORBIDDEN == set()


def test_the_scan_sees_the_rpcs_the_gateway_does_call() -> None:
    assert {
        "SendCommand",
        "ResetEmergencyStop",
        "SetLoadDemand",
        "StreamSystemState",
        "InjectFault",
    } <= attribute_names()
