"""The running service: the OPC UA server, its MQTT bridge and its two projections.

They run side by side until one fails or the service is stopped; then the others are
cancelled, pending gateway sign-outs get a moment to finish, and the server and the
read clients close, in that order.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from cogniboiler_observability import start_metrics_server

from opcua_server.client import AlarmReadClient, PLCStatusClient
from opcua_server.identity import drain_sign_outs
from opcua_server.security import certificate_from_environment
from opcua_server.server import CogniBoilerOPCServer, endpoint_url
from opcua_server.subscriber import MQTTOPCBridge
from opcua_server.upstreams import run_alarm_projection, run_plc_projection

logger = logging.getLogger("opcua_server")
SIGN_OUT_DRAIN_S = 5.0


async def main(args: argparse.Namespace) -> None:
    opc_server = CogniBoilerOPCServer(
        endpoint_url("0.0.0.0", args.opc_port),
        gateway_url=args.gateway_url,
        certificate=certificate_from_environment(),
    )
    alarms_changed = asyncio.Event()
    bridge = MQTTOPCBridge(
        opc_server,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        max_update_hz=args.max_update_hz,
        alarms_changed=alarms_changed,
        mqtt_username=os.environ.get("MQTT_USERNAME", "opcua-server"),
        mqtt_password=os.environ.get("MQTT_PASSWORD") or None,
    )
    plc = PLCStatusClient(args.plc_target)
    alarms = AlarmReadClient(args.alarm_target)

    await opc_server.start()
    start_metrics_server(args.metrics_port, args.metrics_host)
    logger.info(
        "OPC UA server running at %s; mqtt=%s:%d plc=%s alarms=%s gateway=%s",
        opc_server.bound_endpoint,
        args.mqtt_host,
        args.mqtt_port,
        args.plc_target,
        args.alarm_target,
        args.gateway_url,
    )
    tasks = [
        asyncio.ensure_future(work)
        for work in (
            bridge.run(),
            bridge.flush_pending(),
            run_plc_projection(opc_server, plc, args.plc_interval_s),
            run_alarm_projection(
                opc_server, alarms, args.alarm_interval_s, alarms_changed
            ),
        )
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await drain_sign_outs(SIGN_OUT_DRAIN_S)
        await opc_server.stop()
        await plc.close()
        await alarms.close()
