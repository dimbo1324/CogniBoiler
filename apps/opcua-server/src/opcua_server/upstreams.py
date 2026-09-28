"""
PLC and alarm folders, kept current from PLCService and AlarmService.

The PLC status is polled every second; open alarms every few seconds and at once when
alert-manager announces a change on MQTT. While a service cannot be reached, its
communication flag is false and the last values are marked UncertainLastUsableValue.
Each outage is logged once, when it starts, and once more when it ends. Each poll runs
under its own correlation id, so a slow or failing read can be found in the PLC's or
alert-manager's logs.

A defect while projecting a reply is logged once per kind and the loop goes on: the
loops run beside the OPC UA server, and one of them ending would stop the service.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

import grpc
from cogniboiler_observability import correlation_scope
from cogniboiler_runtime import OutageLog

from opcua_server.address_space import (
    ALARM_VARIABLES,
    NODEID_ALARM_COMMUNICATION,
    NODEID_PLC_COMMUNICATION,
    PLC_VARIABLES,
)
from opcua_server.client import AlarmReadClient, PLCStatusClient
from opcua_server.projection import Update, alarm_updates, plc_updates
from opcua_server.server import CogniBoilerOPCServer

logger = logging.getLogger(__name__)


class _Folder[T]:
    """One folder fed by one read: what it reads, how it projects, how it reports."""

    def __init__(
        self,
        opc: CogniBoilerOPCServer,
        what: str,
        read: Callable[[], Awaitable[T]],
        project: Callable[[T], list[Update]],
        flag_node: int,
        node_ids: list[int],
    ) -> None:
        self._opc = opc
        self._what = what
        self._read = read
        self._project = project
        self._flag_node = flag_node
        self._node_ids = [node_id for node_id in node_ids if node_id != flag_node]
        self._outage = OutageLog(logger, what)
        self._reported_defects: set[type[BaseException]] = set()

    async def poll(self) -> None:
        with correlation_scope(None):
            try:
                reply = await self._read()
            except grpc.RpcError as exc:
                if self._outage.failed(exc):
                    await self._guarded(self._mark_outage)
                return
        self._outage.recovered()
        await self._guarded(lambda: self._write(self._project(reply)))

    async def _write(self, updates: list[Update]) -> None:
        for node_id, value in updates:
            await self._opc.update_variable(node_id, value)

    async def _mark_outage(self) -> None:
        await self._opc.update_variable(self._flag_node, False)
        await self._opc.mark_stale(self._node_ids)

    async def _guarded(self, step: Callable[[], Awaitable[None]]) -> None:
        try:
            await step()
        except Exception as exc:
            kind = type(exc)
            if kind in self._reported_defects:
                logger.debug("%s: projection failed again: %r", self._what, exc)
                return
            self._reported_defects.add(kind)
            logger.error("%s: projection failed", self._what, exc_info=exc)


async def run_plc_projection(
    opc: CogniBoilerOPCServer, client: PLCStatusClient, interval_s: float
) -> None:
    folder = _Folder(
        opc,
        "PLCService for the PLC folder",
        client.get_control_status,
        plc_updates,
        NODEID_PLC_COMMUNICATION,
        [variable.node_id for variable in PLC_VARIABLES],
    )
    while True:
        await folder.poll()
        await asyncio.sleep(interval_s)


async def run_alarm_projection(
    opc: CogniBoilerOPCServer,
    client: AlarmReadClient,
    interval_s: float,
    changed: asyncio.Event,
) -> None:
    folder = _Folder(
        opc,
        "AlarmService for the Alarms folder",
        client.open_alarms,
        alarm_updates,
        NODEID_ALARM_COMMUNICATION,
        [variable.node_id for variable in ALARM_VARIABLES],
    )
    while True:
        changed.clear()
        await folder.poll()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(changed.wait(), timeout=interval_s)
