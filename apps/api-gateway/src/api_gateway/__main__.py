"""
API gateway entry point: structured logs, then uvicorn serving the application.

Usage:
    uv run --package api-gateway python -m api_gateway --port 8000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

import uvicorn
from cogniboiler_observability import configure_logging

# aiomqtt needs add_reader(), which the Proactor loop uvicorn picks on Windows lacks;
# uvicorn imports a "module:attr" loop name and calls it to make the loop.
WINDOWS_LOOP = "asyncio:SelectorEventLoop"
# /ws refuses client frames over 8 KiB with its own close code; these stop the server
# from buffering frames of up to 16 MiB, 32 deep, before the handler sees them.
WS_MAX_MESSAGE_BYTES = 64 * 1024
WS_MAX_QUEUED_MESSAGES = 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler API gateway")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1",
        help="proxies whose X-Forwarded-For names the client (the audit log's address)",
    )
    return parser.parse_args(argv)


def event_loop() -> str:
    return WINDOWS_LOOP if sys.platform == "win32" else "auto"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging("api-gateway")
    uvicorn.run(
        "api_gateway.main:app",
        host=args.host,
        port=args.port,
        loop=event_loop(),
        log_config=None,
        server_header=False,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        ws_max_size=WS_MAX_MESSAGE_BYTES,
        ws_max_queue=WS_MAX_QUEUED_MESSAGES,
    )


if __name__ == "__main__":
    main()
