"""Take the README's screenshots and its demo GIF from a running stack.

Runs the Playwright spec in apps/web/readme/ against the console — it plays the demo at
ten times real speed through the gateway and leaves the unit at nominal in real time —
then writes indexed-colour PNGs and one looping GIF into docs/images/, each inside the
repository's large-file limit. ``--skip-capture`` re-encodes the last capture without
playing the demo again, for changing a size or a palette.

The encoding needs Pillow, which only the project environment has, so the script hands
itself to ``uv run`` when it is started by another interpreter.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from scripts._toolkit.config import (
    ScriptConfigError,
    load_config,
    repo_root,
    resolve_inside,
)
from scripts._toolkit.console import fail, heading, ok
from scripts._toolkit.processes import NOT_FOUND, run
from scripts._toolkit.reexec import has_module, reexec_under_uv
from scripts.readme_media.media import (
    BYTES_PER_KIB,
    MediaError,
    durations,
    encode_gif,
    encode_png,
    first_that_fits,
    load_attempts,
    load_frames,
    select_frames,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def capture_commands(config: dict[str, Any], install: bool) -> list[list[str]]:
    web = str(config["web_dir"])
    commands = []
    if install:
        commands.append(
            ["pnpm", "--dir", web, "exec", "playwright", "install", "chromium"]
        )
    commands.append(
        [
            "pnpm",
            "--dir",
            web,
            "exec",
            "playwright",
            "test",
            "--config",
            str(config["playwright_config"]),
        ]
    )
    return commands


def capture(root: Path, config: dict[str, Any], url: str, install: bool) -> bool:
    if not (root / str(config["web_dir"]) / "node_modules").is_dir():
        fail("apps/web/node_modules is missing — run: pnpm --dir apps/web install")
        return False
    try:
        raw = resolve_inside(root, str(config["raw_dir"]))
    except ScriptConfigError as error:
        fail(f"raw_dir in readme_media.json: {error}")
        return False
    # Only this script writes here, and git ignores it: an old frame left behind would
    # end up in the next GIF.
    try:
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir(parents=True)
    except OSError as error:
        fail(f"cannot empty the capture folder {raw}: {error}")
        return False
    env = {"CONSOLE_URL": url, "README_MEDIA_DIR": str(raw)}
    for command in capture_commands(config, install):
        result = run(command, root, env=env)
        if result.returncode == NOT_FOUND:
            fail("pnpm is not on PATH — install Node and enable corepack")
            return False
        if not result.ok:
            fail(
                "the capture did not finish; the unit is put back at nominal either way"
            )
            return False
    return True


def encode(root: Path, config: dict[str, Any]) -> None:
    raw = resolve_inside(root, str(config["raw_dir"]))
    out = resolve_inside(root, str(config["out_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    budget = int(config["budget_kib"]) * BYTES_PER_KIB

    attempts = load_attempts(config["screenshot_attempts"])
    for shot in config["screenshots"]:
        source = raw / str(shot["source"])
        if not source.is_file():
            raise MediaError(f"{source.name} was not captured")
        attempt, data = first_that_fits(
            attempts, lambda chosen, path=source: encode_png(path, chosen), budget
        )
        (out / str(shot["target"])).write_bytes(data)
        ok(
            f"{shot['target']}: {len(data) / BYTES_PER_KIB:.0f} KiB, {attempt.describe()}"
        )

    gif = config["gif"]
    frames = load_frames((raw / str(gif["manifest"])).read_text(encoding="utf-8"))
    chosen_frames = select_frames(frames, int(gif["max_frames"]))
    shown = durations(chosen_frames, int(gif["frame_ms"]), int(gif["last_hold_ms"]))
    sources = [raw / frame.file for frame in chosen_frames]
    missing = [path.name for path in sources if not path.is_file()]
    if missing:
        raise MediaError(f"frames named in the manifest are missing: {missing[:3]}")
    attempt, data = first_that_fits(
        load_attempts(gif["attempts"]),
        lambda chosen: encode_gif(sources, shown, chosen),
        budget,
    )
    (out / str(gif["target"])).write_bytes(data)
    ok(
        f"{gif['target']}: {len(data) / BYTES_PER_KIB:.0f} KiB, {attempt.describe()}, "
        f"{len(chosen_frames)} of {len(frames)} frames, {sum(shown) / 1000:.0f} s"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="readme-media",
        description="Take the README's screenshots and demo GIF from a running stack.",
    )
    parser.add_argument("--url", help="console URL (default: the stack's nginx entry)")
    parser.add_argument(
        "--no-install", action="store_true", help="do not install Chromium first"
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="re-encode the last capture instead of playing the demo again",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    if not has_module("PIL"):
        code = reexec_under_uv("scripts.readme_media", argv, root)
        if code is not None:
            return code
        fail("Pillow is unavailable — run: uv sync --all-packages")
        return 1

    config = load_config(SCRIPT_DIR, "readme_media.json")
    url = args.url or str(config["url"])
    heading(f"readme-media — {url}")
    if not args.skip_capture and not capture(root, config, url, not args.no_install):
        return 1
    try:
        encode(root, config)
    except (MediaError, ScriptConfigError) as error:
        fail(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
