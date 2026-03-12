#!/usr/bin/env python3
"""Generate Python gRPC stubs from .proto files.

Run from repo root:
    uv run python scripts/generate_proto.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROTO_DIR = REPO_ROOT / "shared" / "proto"
OUT_DIR = REPO_ROOT / "shared" / "generated"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create __init__.py if missing
init = OUT_DIR / "__init__.py"
if not init.exists():
    init.write_text('"""Auto-generated gRPC stubs. Do not edit manually."""\n')

proto_files = list(PROTO_DIR.glob("*.proto"))
if not proto_files:
    print("No .proto files found in", PROTO_DIR)
    sys.exit(1)

cmd = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    f"--proto_path={PROTO_DIR}",
    f"--python_out={OUT_DIR}",
    f"--grpc_python_out={OUT_DIR}",
    *[str(p.name) for p in proto_files],
]


print("Running:", " ".join(cmd))
result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

if result.returncode != 0:
    print("ERROR:", result.stderr)
    sys.exit(1)

print("Generated stubs in", OUT_DIR)
for f in sorted(OUT_DIR.glob("*.py")):
    print(" ", f.name)
