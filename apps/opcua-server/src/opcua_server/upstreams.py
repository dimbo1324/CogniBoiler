"""
PLC and alarm folders, kept current from PLCService and AlarmService.

The PLC status is polled every second; open alarms every few seconds and at once when
alert-manager announces a change on MQTT. While a service cannot be reached, its
communication flag is false and the last values are marked UncertainLastUsableValue.
Each outage is logged once, when it starts, and once more when it ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import grpc

from opcua_server.address_space import ALARM_VARIABLES, PLC_VARIABLES
from opcua_server.client import AlarmReadClient, PLCStatusClient
from opcua_server.projection import Update, alarm_updates, plc_updates
from opcua_server.server import CogniBoilerOPCServer

logger = logging.getLogger(__name__)

NODEID_PLC_COMMUNICATION = 2712
NODEID_ALARM_COMMUNICATION = 2804


async def _write(opc: CogniBoilerOPCServer, updates: list[Update]) -> None:
    for node_id, value in updates:
        await opc.update_variable(node_id, value)


async def _outage(
    opc: CogniBoilerOPCServer, flag_node: int, node_ids: list[int]
) -> None:
    await opc.update_variable(flag_node, False)
    await opc.mark_stale(node_id for node_id in node_ids if node_id != flag_node)


async def run_plc_projection(
    opc: CogniBoilerOPCServer, client: PLCStatusClient, interval_s: float
) -> None:
    node_ids = [variable.node_id for variable in PLC_VARIABLES]
    down = False
    while True:
        try:
            status = await client.get_control_status()
        except grpc.RpcError as exc:
            if not down:
                logger.warning("PLCService unreachable for the PLC folder: %s", exc)
                await _outage(opc, NODEID_PLC_COMMUNICATION, node_ids)
                down = True
        else:
            if down:
                logger.info("PLCService reachable again")
                down = False
            await _write(opc, plc_updates(status))
        await asyncio.sleep(interval_s)


async def run_alarm_projection(
    opc: CogniBoilerOPCServer,
    client: AlarmReadClient,
    interval_s: float,
    changed: asyncio.Event,
) -> None:
    node_ids = [variable.node_id for variable in ALARM_VARIABLES]
    down = False
    while True:
        changed.clear()
        try:
            alarms = await client.open_alarms()
        except grpc.RpcError as exc:
            if not down:
                logger.warning(
                    "AlarmService unreachable for the Alarms folder: %s", exc
                )
                await _outage(opc, NODEID_ALARM_COMMUNICATION, node_ids)
                down = True
        else:
            if down:
                logger.info("AlarmService reachable again")
                down = False
            await _write(opc, alarm_updates(alarms))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), timeout=interval_s)
