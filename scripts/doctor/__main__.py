"""Report what this machine can and cannot do — read-only, always exit-code honest.

What it probes lives in ``config/tools.json``; adding a tool is a JSON edit.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok, step, warn
from scripts._toolkit.processes import TIMED_OUT, capture, find_tool

SCRIPT_DIR = Path(__file__).resolve().parent


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no version output)"


def _check_tools(
    config: dict[str, list[dict[str, object]]], root: Path
) -> tuple[list[str], list[str]]:
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for entry in config["tools"]:
        name = str(entry["name"])
        required = bool(entry["required"])
        resolved = find_tool(name)
        if resolved is None:
            (missing_required if required else missing_optional).append(name)
            label = "required" if required else "optional"
            warn(f"{name:<9} MISSING ({label}) — needed for {entry['needed_for']}")
            continue
        version_args = [str(arg) for arg in entry["version_args"]]  # type: ignore[union-attr]
        code, output = capture([name, *version_args], root)
        if code == TIMED_OUT:
            warn(f"{name:<9} did not answer in time ({resolved})")
            continue
        if code != 0:
            warn(
                f"{name:<9} found but `{name} {' '.join(version_args)}` failed ({resolved})"
            )
            continue
        ok(f"{name:<9} {_first_line(output)}")
    return missing_required, missing_optional


def _check_python_pin(root: Path) -> None:
    pin_file = root / ".python-version"
    if not pin_file.exists():
        warn(".python-version is missing — uv cannot know which interpreter to use")
        return
    pinned = pin_file.read_text(encoding="utf-8").strip()
    code, output = capture(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "import sys; print(sys.version.split()[0])",
        ],
        root,
    )
    if code != 0:
        warn(
            f"project environment unusable (pinned {pinned}) — run: uv sync --all-packages"
        )
        return
    actual = _first_line(output)
    if actual.startswith(pinned):
        ok(f"project environment runs Python {actual} (pinned {pinned})")
    else:
        warn(
            f"project environment runs Python {actual}, but {pinned} is pinned — run: uv sync --all-packages"
        )


def _check_docker_daemon(root: Path) -> None:
    if find_tool("docker") is None:
        return
    code, output = capture(["docker", "info", "--format", "{{.ServerVersion}}"], root)
    if code == 0:
        ok(f"docker daemon answers (engine {_first_line(output)})")
    else:
        warn(
            "docker is installed but the daemon does not answer — start Docker Desktop"
        )


def _check_git_settings(root: Path) -> list[str]:
    problems: list[str] = []
    hook = root / ".git" / "hooks" / "pre-commit"
    if hook.exists() and "pre-commit" in hook.read_text(
        encoding="utf-8", errors="replace"
    ):
        ok("pre-commit hook installed")
    else:
        warn("pre-commit hook NOT installed — commits will not be formatted")
        info("run: python dev_tools_scripts_runner.py install-hooks")
        problems.append("pre-commit hook")

    if os.name == "nt":
        code, output = capture(["git", "config", "--get", "core.longpaths"], root)
        if code == 0 and output.strip() == "true":
            ok("core.longpaths true")
        else:
            warn("core.longpaths not enabled — deep node_modules paths may break git")
            info("run: git config core.longpaths true")
            problems.append("core.longpaths")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="Check that the tools the other scripts need are present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when an optional tool is missing",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "tools.json")

    heading("CogniBoiler — environment check")
    info(f"repository   {root}")
    info(f"python       {sys.version.split()[0]}  ({sys.executable})")
    info(f"platform     {platform.platform()}")

    step("tools")
    missing_required, missing_optional = _check_tools(config, root)

    step("project environment")
    _check_python_pin(root)
    _check_docker_daemon(root)

    step("paths")
    for entry in config["paths"]:
        if (root / str(entry["path"])).exists():
            ok(str(entry["label"]))
        else:
            warn(f"{entry['label']} missing — {entry['hint']}")

    step("git settings")
    git_problems = _check_git_settings(root)

    heading("verdict")
    if missing_required:
        fail(f"missing required tool(s): {', '.join(missing_required)}")
        return 1
    if missing_optional and args.strict:
        fail(f"missing optional tool(s) with --strict: {', '.join(missing_optional)}")
        return 1
    if missing_optional:
        warn(f"optional tool(s) absent: {', '.join(missing_optional)}")
        info("Not a failure — the affected steps say so when they run.")
    if git_problems:
        warn(f"git setting(s) to fix: {', '.join(git_problems)}")
    ok("environment is usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
