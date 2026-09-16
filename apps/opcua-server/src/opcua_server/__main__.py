"""
OPC UA Server entry point.

Usage:
    uv run --package opcua-server python -m opcua_server
    uv run --package opcua-server python -m opcua_server --mqtt-host localhost --opc-port 4840
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

from opcua_server.client import AlarmReadClient, PLCStatusClient
from opcua_server.server import CogniBoilerOPCServer
from opcua_server.subscriber import MQTTOPCBridge
from opcua_server.upstreams import run_alarm_projection, run_plc_projection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("asyncua").setLevel(logging.WARNING)
logger = logging.getLogger("opcua_server")


async def main(args: argparse.Namespace) -> None:
    opc_server = CogniBoilerOPCServer(
        endpoint=f"opc.tcp://0.0.0.0:{args.opc_port}/cogniboiler",
        gateway_url=args.gateway_url,
    )
    alarms_changed = asyncio.Event()
    bridge = MQTTOPCBridge(
        opc_server,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        max_update_hz=args.max_update_hz,
        alarms_changed=alarms_changed,
    )
    plc = PLCStatusClient(args.plc_target)
    alarms = AlarmReadClient(args.alarm_target)

    await opc_server.start()
    logger.info(
        "OPC UA server running at opc.tcp://localhost:%d/cogniboiler; mqtt=%s:%d "
        "plc=%s alarms=%s gateway=%s",
        args.opc_port,
        args.mqtt_host,
        args.mqtt_port,
        args.plc_target,
        args.alarm_target,
        args.gateway_url,
    )

    async def log_stats() -> None:
        while True:
            await asyncio.sleep(30)
            stats = bridge.stats
            logger.info(
                "Bridge stats: received=%d mapped=%d skipped=%d",
                stats["received"],
                stats["mapped"],
                stats["skipped"],
            )

    try:
        await asyncio.gather(
            bridge.run(),
            bridge.flush_pending(),
            run_plc_projection(opc_server, plc, args.plc_interval_s),
            run_alarm_projection(
                opc_server, alarms, args.alarm_interval_s, alarms_changed
            ),
            log_stats(),
        )
    finally:
        await opc_server.stop()
        await plc.close()
        await alarms.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler OPC UA Server")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--opc-port", type=int, default=4840)
    parser.add_argument("--plc-target", default="localhost:50051")
    parser.add_argument("--alarm-target", default="localhost:50053")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--max-update-hz", type=float, default=5.0)
    parser.add_argument("--plc-interval-s", type=float, default=1.0)
    parser.add_argument("--alarm-interval-s", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(
        main(parse_args()),
        # aiomqtt needs add_reader(), which the Windows proactor loop does not have.
        loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None,
    )
