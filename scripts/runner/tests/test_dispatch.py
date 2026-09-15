"""What the orchestrator does with its argv, and where the interface language comes from.

Both were changed after review found them wrong, and both were verified by hand — in a
task whose own lesson was that unverified logic under `scripts/` reaches `main` green.
These are the tests that make the gate able to check them.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.runner.config_loader import CONFIG_DIR, ConfigLoader
from scripts.runner.exceptions import ConfigValidationError
from scripts.runner.main import CliApp

ROOT = Path(__file__).resolve().parents[2].parent


class RecordingRunner:
    """Stands in for `ScriptRunner` so dispatch can be asserted without launching a
    subprocess — the point is which script was chosen, not what it does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, script, argv: list[str]) -> int:
        self.calls.append((script.title, list(argv)))
        return 0


class DispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = CliApp(ROOT)
        self.runner = RecordingRunner()
        self.app.runner = self.runner

    def test_a_named_script_runs_with_its_own_arguments(self) -> None:
        self.assertEqual(self.app.main(["doctor", "--verbose"]), 0)
        self.assertEqual(self.runner.calls, [("doctor", ["--verbose"])])

    def test_an_alias_resolves_to_the_same_script(self) -> None:
        self.app.main(["gate"])
        self.assertEqual(self.runner.calls[0][0], "quality-gate")

    def test_leading_flags_reach_the_default_script(self) -> None:
        """`... --quick` without naming quality-gate is the reason this path exists."""
        self.assertEqual(self.app.main(["--quick"]), 0)
        self.assertEqual(self.runner.calls, [("quality-gate", ["--quick"])])

    def test_a_misspelled_script_name_is_an_error_not_the_default_script(self) -> None:
        """`gat` used to run the entire quality gate. A typo must not be
        indistinguishable from a deliberate choice."""
        self.assertEqual(self.app.main(["gat"]), 2)
        self.assertEqual(self.runner.calls, [])

    def test_list_and_help_do_not_launch_anything(self) -> None:
        for argv in (["list"], ["help"], ["--help"]):
            with self.subTest(argv=argv), redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(self.app.main(argv), 0)
                self.assertIn("quality-gate", captured.getvalue())
        self.assertEqual(self.runner.calls, [])

    def test_an_unknown_script_name_is_reported_on_stderr(self) -> None:
        with redirect_stderr(io.StringIO()) as captured:
            self.app.main(["gat"])
        self.assertIn("gat", captured.getvalue())


class DefaultLangTest(unittest.TestCase):
    """`default_lang` was declared in meta.json, required by the loader, and read by
    nobody: the language came from a Python constant, so editing the JSON did nothing."""

    def load_with_lang(self, value: str):
        catalog = {
            p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in CONFIG_DIR.glob("*.json")
        }
        catalog["meta.json"]["default_lang"] = value
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for name, payload in catalog.items():
                (directory / name).write_text(json.dumps(payload), encoding="utf-8")
            return ConfigLoader(ROOT, directory).load()

    def test_the_configured_language_reaches_the_registry(self) -> None:
        for value in ("en", "ru"):
            with self.subTest(value=value):
                self.assertEqual(self.load_with_lang(value).default_lang, value)

    def test_an_unsupported_language_is_rejected_by_name(self) -> None:
        with self.assertRaises(ConfigValidationError) as caught:
            self.load_with_lang("de")
        self.assertIn("default_lang", str(caught.exception))

    def test_help_output_follows_the_configured_language(self) -> None:
        app = CliApp(ROOT)
        app.registry = self.load_with_lang("ru")
        self.assertEqual(app.parse_help_args([]), ("ru", []))
        self.assertEqual(app.parse_help_args(["--lang", "en"]), ("en", []))


if __name__ == "__main__":
    unittest.main()
