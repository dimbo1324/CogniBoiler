"""Regenerate the Python gRPC stubs in shared/generated from shared/proto, or check them.

The stubs are committed, so a contract change and its regenerated stubs land in one
commit. ``--check`` generates into a temporary directory and fails when the committed
files differ — how the gate notices a .proto edited without regenerating.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, info, ok
from scripts._toolkit.reexec import has_module, reexec_under_uv

SCRIPT_DIR = Path(__file__).resolve().parent
STUB_GLOB = "*_pb2*.py"


def generate(proto_dir: Path, out_dir: Path) -> list[Path]:
    protos = sorted(proto_dir.glob("*.proto"))
    if not protos:
        raise FileNotFoundError(f"no .proto files in {proto_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        *(proto.name for proto in protos),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "grpc_tools.protoc failed")
    return sorted(out_dir.glob(STUB_GLOB))


def _normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def drifted(fresh_dir: Path, committed_dir: Path) -> list[str]:
    names = {path.name for path in fresh_dir.glob(STUB_GLOB)}
    names |= {path.name for path in committed_dir.glob(STUB_GLOB)}
    differing = []
    for name in sorted(names):
        fresh, committed = fresh_dir / name, committed_dir / name
        if not fresh.exists() or not committed.exists():
            differing.append(name)
        elif _normalized(fresh) != _normalized(committed):
            differing.append(name)
    return differing


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="generate-proto",
        description="Regenerate shared/generated from shared/proto, or check it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed stubs differ from freshly generated ones",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "proto.json")

    if not has_module("grpc_tools"):
        code = reexec_under_uv("scripts.generate_proto", argv, root)
        if code is not None:
            return code
        fail("grpc_tools is unavailable — run: uv sync --all-packages")
        return 1

    proto_dir = root / str(config["proto_dir"])
    out_dir = root / str(config["output_dir"])

    try:
        if args.check:
            with tempfile.TemporaryDirectory() as temp:
                generate(proto_dir, Path(temp))
                differing = drifted(Path(temp), out_dir)
            if differing:
                fail(
                    f"stubs out of date: {', '.join(differing)} — run: "
                    "python dev_tools_scripts_runner.py generate-proto"
                )
                return 1
            ok("committed stubs match shared/proto")
            return 0

        for path in generate(proto_dir, out_dir):
            info(str(path.relative_to(root)))
    except (FileNotFoundError, RuntimeError) as error:
        fail(str(error))
        return 1
    ok(f"stubs regenerated in {config['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
