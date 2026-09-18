"""Export the gateway's OpenAPI schema and the console's types generated from it, or check them.

The schema in ``shared/openapi`` is the REST contract between the gateway and the console;
the console's TypeScript types are generated from it, never written by hand. Both are
committed. ``--check`` rebuilds them in memory and fails when either differs — how the gate
notices a route changed without its contract, or a contract changed without the types.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, info, ok, warn
from scripts._toolkit.processes import find_tool
from scripts._toolkit.reexec import has_module, reexec_under_uv

SCRIPT_DIR = Path(__file__).resolve().parent


def export_schema(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    generated = str(root / str(config["grpc_stubs_dir"]))
    if generated not in sys.path:
        sys.path.insert(0, generated)
    from api_gateway.main import create_app

    schema: dict[str, Any] = create_app().openapi()
    return schema


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def _normalized(text: str) -> str:
    return text.replace("\r\n", "\n")


def generate_types(root: Path, config: dict[str, Any], schema: Path, out: Path) -> None:
    pnpm = find_tool("pnpm")
    if pnpm is None:
        raise RuntimeError("pnpm is not on PATH; it generates the console types")
    command = [
        pnpm,
        "--dir",
        str(root / str(config["web_dir"])),
        "exec",
        "openapi-typescript",
        str(schema),
        "--output",
        str(out),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "openapi-typescript failed"
        )


def _frontend_ready(root: Path, config: dict[str, Any]) -> bool:
    return (root / str(config["web_dir"]) / "node_modules").is_dir()


def check(root: Path, config: dict[str, Any]) -> int:
    schema_path = root / str(config["schema"])
    types_path = root / str(config["types"])
    fresh = render(export_schema(root, config))
    committed = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""
    if _normalized(committed) != fresh:
        fail(
            f"{config['schema']} does not match the gateway's routes — run: "
            "python dev_tools_scripts_runner.py generate-openapi"
        )
        return 1
    ok(f"{config['schema']} matches the gateway")

    if not _frontend_ready(root, config):
        if os.environ.get("CI"):
            fail(
                "apps/web/node_modules is missing; the console types cannot be checked"
            )
            return 1
        warn("console types not checked: apps/web/node_modules is missing")
        return 0
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / "schema.gen.ts"
        generate_types(root, config, schema_path, out)
        fresh_types = out.read_text(encoding="utf-8")
    committed_types = (
        types_path.read_text(encoding="utf-8") if types_path.exists() else ""
    )
    if _normalized(committed_types) != _normalized(fresh_types):
        fail(
            f"{config['types']} is out of date — run: "
            "python dev_tools_scripts_runner.py generate-openapi"
        )
        return 1
    ok(f"{config['types']} matches {config['schema']}")
    return 0


def write(root: Path, config: dict[str, Any]) -> int:
    schema_path = root / str(config["schema"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        render(export_schema(root, config)), encoding="utf-8", newline="\n"
    )
    info(str(config["schema"]))
    if not _frontend_ready(root, config):
        warn("console types not generated: run pnpm --dir apps/web install first")
        return 1
    types_path = root / str(config["types"])
    generate_types(root, config, schema_path, types_path)
    types_path.write_text(
        _normalized(types_path.read_text(encoding="utf-8")),
        encoding="utf-8",
        newline="\n",
    )
    info(str(config["types"]))
    ok("OpenAPI schema and console types regenerated")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="generate-openapi",
        description="Export the gateway's OpenAPI schema and the console types, or check them.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed schema or types differ from fresh ones",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "openapi.json")

    if not has_module("api_gateway"):
        code = reexec_under_uv("scripts.generate_openapi", argv, root)
        if code is not None:
            return code
        fail("the api-gateway package is unavailable — run: uv sync --all-packages")
        return 1

    try:
        return check(root, config) if args.check else write(root, config)
    except RuntimeError as error:
        fail(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
