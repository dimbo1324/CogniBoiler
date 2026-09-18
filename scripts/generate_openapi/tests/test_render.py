"""How generate-openapi writes and compares the schema.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import json
import unittest

from scripts.generate_openapi.__main__ import _normalized, render


class RenderTest(unittest.TestCase):
    def test_the_schema_is_indented_json_ending_in_a_newline(self) -> None:
        text = render({"openapi": "3.1.0", "info": {"title": "Gateway — test"}})
        self.assertTrue(text.endswith("}\n"))
        self.assertIn('\n  "info": {', text)
        self.assertIn("Gateway — test", text)
        self.assertEqual(json.loads(text)["openapi"], "3.1.0")

    def test_line_endings_do_not_count_as_drift(self) -> None:
        self.assertEqual(_normalized("a\r\nb\r\n"), "a\nb\n")


if __name__ == "__main__":
    unittest.main()
