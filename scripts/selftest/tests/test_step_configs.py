"""Every step a script runs through ``uv run`` starts a module, not a console script.

On the owner's machine Windows App Control refuses to spawn ``ruff.exe``, ``pytest.exe``
and friends from ``.venv/Scripts`` at random, while ``python -m <tool>`` is allowed
(.ai/project/14-command-reference.md, platform notes). The gate was moved to modules;
a step list that still names an executable fails the same way.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[2]
UV_RUN = ["uv", "run", "--no-sync"]


def step_argvs(scripts: Path) -> Iterator[tuple[str, list[str]]]:
    """Every ``argv`` list in any script's config, with the file it came from."""
    for path in sorted(scripts.glob("*/config/*.json")):
        stack: list[Any] = [json.loads(path.read_text(encoding="utf-8"))]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                argv = node.get("argv")
                if isinstance(argv, list):
                    yield path.relative_to(scripts).as_posix(), [str(a) for a in argv]
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)


class UvRunStepTest(unittest.TestCase):
    def test_the_configs_hold_uv_run_steps_at_all(self) -> None:
        self.assertTrue(
            any(argv[:3] == UV_RUN for _, argv in step_argvs(SCRIPTS)),
            "found no uv run step: the scan is not seeing the configs",
        )

    def test_every_uv_run_step_starts_python(self) -> None:
        for source, argv in step_argvs(SCRIPTS):
            if argv[:3] != UV_RUN:
                continue
            with self.subTest(source=source, argv=argv):
                self.assertEqual(argv[3], "python")


if __name__ == "__main__":
    unittest.main()
