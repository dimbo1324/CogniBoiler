"""The catalog loader refuses a hand-edit of the wrong type instead of guessing.

``tuple("qg")`` is ``("q", "g")``: an ``"aliases": "qg"`` string registered the one-letter
identifiers ``q`` and ``g``, so a typo launched a script instead of reporting an unknown
name. These tests load a copy of the shipped catalog with one field broken.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.runner.config_loader import CONFIG_DIR, ConfigLoader
from scripts.runner.exceptions import ConfigValidationError

ROOT = Path(__file__).resolve().parents[2].parent


def load_with(first_script: dict[str, Any]) -> None:
    """Load the shipped catalog with its first script entry updated by ``first_script``."""
    catalog = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in CONFIG_DIR.glob("*.json")
    }
    catalog["scripts.json"][0].update(first_script)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for name, payload in catalog.items():
            (directory / name).write_text(json.dumps(payload), encoding="utf-8")
        ConfigLoader(ROOT, directory).load()


class FieldTypeTest(unittest.TestCase):
    def test_the_shipped_catalog_passes(self) -> None:
        load_with({})

    def test_a_list_field_given_as_a_string_is_refused(self) -> None:
        for key in ("aliases", "examples", "platforms"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ConfigValidationError, key):
                    load_with({key: "qg"})

    def test_a_list_holding_an_empty_or_non_string_item_is_refused(self) -> None:
        for value in (["gate", ""], ["gate", 7]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigValidationError, "aliases"):
                    load_with({"aliases": value})

    def test_destructive_must_be_a_real_boolean(self) -> None:
        with self.assertRaisesRegex(ConfigValidationError, "destructive"):
            load_with({"destructive": "false"})

    def test_a_title_must_be_a_non_empty_string(self) -> None:
        for value in (42, ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigValidationError, "title"):
                    load_with({"title": value})

    def test_both_languages_of_a_text_must_be_strings(self) -> None:
        with self.assertRaisesRegex(ConfigValidationError, "summary"):
            load_with({"summary": {"en": "Run it.", "ru": None}})


if __name__ == "__main__":
    unittest.main()
