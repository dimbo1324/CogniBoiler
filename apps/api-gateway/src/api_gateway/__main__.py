"""
API gateway entry point: structured logs, then uvicorn serving the application.

Usage:
    uv run --package api-gateway python -m api_gateway --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "shared" / "generated"))

import uvicorn
from cogniboiler_observability import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogniBoiler API gateway")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging("api-gateway")
    uvicorn.run(
        "api_gateway.main:app",
        host=args.host,
        port=args.port,
        log_config=None,
        server_header=False,
    )
