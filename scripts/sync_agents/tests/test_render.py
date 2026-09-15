"""How AGENTS.md is assembled from the rule modules.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sync_agents.render import (
    Module,
    SyncError,
    collect_modules,
    render,
    size_kib,
)

BANNER = "<!-- banner -->"


class RenderTest(unittest.TestCase):
    def test_an_inline_module_is_embedded_in_full(self) -> None:
        rendered = render(
            [Module(".ai/universal/01-a.md", "# Alpha\n\nBody text.")], BANNER
        )
        self.assertIn("Body text.", rendered)
        self.assertIn("<!-- module: .ai/universal/01-a.md -->", rendered)
        self.assertTrue(rendered.startswith(BANNER))
        self.assertTrue(rendered.endswith("\n"))

    def test_an_extended_module_contributes_only_its_essence(self) -> None:
        body = (
            "<!-- tier: extended -->\n\n# Beta\n\n"
            "> **Essence.** Short summary.\n\nLong body."
        )
        rendered = render([Module(".ai/universal/02-b.md", body)], BANNER)
        self.assertIn("Short summary.", rendered)
        self.assertIn("File: `.ai/universal/02-b.md`", rendered)
        self.assertNotIn("Long body.", rendered)

    def test_extended_modules_follow_every_inline_module(self) -> None:
        modules = [
            Module(
                ".ai/universal/01-a.md",
                "<!-- tier: extended -->\n# A\n> **Essence.** a",
            ),
            Module(".ai/project/10-b.md", "# B\n\nInline body."),
        ]
        rendered = render(modules, BANNER)
        self.assertLess(
            rendered.index("Inline body."), rendered.index("Modules loaded on demand")
        )

    def test_title_and_essence_are_extracted(self) -> None:
        module = Module(
            "x.md",
            "<!-- tier: extended -->\n\n# Gamma\n\n> **Essence.** Keep it tight.",
        )
        self.assertEqual(module.title, "Gamma")
        self.assertEqual(module.essence, "Keep it tight.")
        self.assertTrue(module.is_extended)

    def test_a_marker_deep_in_the_body_does_not_make_a_module_extended(self) -> None:
        body = "# Delta\n\n1\n2\n3\n4\n5\n<!-- tier: extended -->"
        self.assertFalse(Module("x.md", body).is_extended)

    def test_size_counts_bytes_not_characters(self) -> None:
        self.assertAlmostEqual(size_kib("я" * 512), 1.0)


class CollectTest(unittest.TestCase):
    def test_modules_are_read_in_group_then_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = (
                ("universal", "02-b.md"),
                ("universal", "01-a.md"),
                ("project", "10-c.md"),
            )
            for group, name in layout:
                directory = root / ".ai" / group
                directory.mkdir(parents=True, exist_ok=True)
                (directory / name).write_bytes(f"# {name}\r\n\r\nbody\r\n".encode())
            modules = collect_modules(root, ["universal", "project"])
        self.assertEqual(
            [module.relative_path for module in modules],
            [".ai/universal/01-a.md", ".ai/universal/02-b.md", ".ai/project/10-c.md"],
        )
        self.assertNotIn("\r", modules[0].body)

    def test_a_missing_group_is_an_error_naming_it(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaises(SyncError) as caught,
        ):
            collect_modules(Path(temp), ["universal"])
        self.assertIn("universal", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
