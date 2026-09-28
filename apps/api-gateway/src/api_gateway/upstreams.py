"""
The upstream clients a route can ask for, typed.

The lifespan puts one client per upstream on the application state; a route declares
the one it talks to as a parameter (`plc: PlcClient`) instead of reading the untyped
state itself. Tests put fakes on the same state.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from api_gateway.clients import (
    AlarmGatewayClient,
    PhysicsGatewayClient,
    PLCGatewayClient,
)
from api_gateway.historian_query import HistorianQueryClient

PHYSICS_SERVICE = "PhysicsService"
PLC_SERVICE = "PLCService"
ALARM_SERVICE = "AlarmService"


def get_physics_client(request: Request) -> PhysicsGatewayClient:
    return cast(PhysicsGatewayClient, request.app.state.physics_client)


def get_plc_client(request: Request) -> PLCGatewayClient:
    return cast(PLCGatewayClient, request.app.state.plc_client)


def get_alarm_client(request: Request) -> AlarmGatewayClient:
    return cast(AlarmGatewayClient, request.app.state.alarm_client)


def get_historian_client(request: Request) -> HistorianQueryClient:
    return cast(HistorianQueryClient, request.app.state.historian_client)


PhysicsClient = Annotated[PhysicsGatewayClient, Depends(get_physics_client)]
PlcClient = Annotated[PLCGatewayClient, Depends(get_plc_client)]
AlarmClient = Annotated[AlarmGatewayClient, Depends(get_alarm_client)]
HistorianClient = Annotated[HistorianQueryClient, Depends(get_historian_client)]
