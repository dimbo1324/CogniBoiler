"""The dotenv subset dev-secrets reads and writes, and its never-overwrite merge.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest

from scripts._toolkit.envfile import (
    EnvFileError,
    format_value,
    merge,
    parse,
    values,
)

PEM = "-----BEGIN PUBLIC KEY-----\nABC\nDEF\n-----END PUBLIC KEY-----"


class ParseTest(unittest.TestCase):
    def test_assignments_comments_and_blank_lines(self) -> None:
        entries = parse("# heading\n\nUSER=cogniboiler\nPASSWORD=\n")
        self.assertEqual(values(entries), {"USER": "cogniboiler", "PASSWORD": ""})
        self.assertEqual(entries[0].raw, "# heading")

    def test_a_quoted_value_may_span_lines(self) -> None:
        text = f'KEY="{PEM}"\nNEXT=1\n'
        self.assertEqual(values(parse(text)), {"KEY": PEM, "NEXT": "1"})

    def test_windows_line_endings_are_accepted(self) -> None:
        text = f'KEY="{PEM}"\r\nNEXT=1\r\n'.replace("\n", "\r\n").replace("\r\r", "\r")
        self.assertEqual(values(parse(text))["KEY"], PEM)

    def test_an_unclosed_quote_is_an_error(self) -> None:
        with self.assertRaises(EnvFileError):
            parse('KEY="never closed\n')

    def test_a_line_that_is_not_an_assignment_is_an_error(self) -> None:
        with self.assertRaises(EnvFileError):
            parse("just words\n")


class FormatTest(unittest.TestCase):
    def test_plain_values_stay_unquoted(self) -> None:
        self.assertEqual(format_value("abc-123_xyz"), "abc-123_xyz")

    def test_multiline_values_are_quoted_and_round_trip(self) -> None:
        rendered = f"KEY={format_value(PEM)}\n"
        self.assertEqual(values(parse(rendered))["KEY"], PEM)

    def test_a_double_quote_is_refused(self) -> None:
        with self.assertRaises(EnvFileError):
            format_value('a"b')


class MergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.template = parse(
            "# db\nPOSTGRES_USER=cogniboiler\nPOSTGRES_PASSWORD=\nOPTIONAL_SETTING=\n"
        )
        self.calls: list[str] = []

    def _generate(self) -> str:
        self.calls.append("generated")
        return "fresh-secret"

    def test_an_existing_value_is_never_overwritten(self) -> None:
        text, generated, kept = merge(
            self.template,
            {"POSTGRES_PASSWORD": "mine"},
            {"POSTGRES_PASSWORD": self._generate},
        )
        self.assertIn("POSTGRES_PASSWORD=mine", text)
        self.assertEqual(generated, [])
        self.assertEqual(kept, ["POSTGRES_PASSWORD"])
        self.assertEqual(self.calls, [])

    def test_an_empty_secret_is_generated(self) -> None:
        text, generated, _ = merge(
            self.template, {}, {"POSTGRES_PASSWORD": self._generate}
        )
        self.assertIn("POSTGRES_PASSWORD=fresh-secret", text)
        self.assertEqual(generated, ["POSTGRES_PASSWORD"])

    def test_template_defaults_and_comments_are_carried_over(self) -> None:
        text, _, _ = merge(self.template, {}, {})
        self.assertTrue(text.startswith("# db\nPOSTGRES_USER=cogniboiler\n"))
        self.assertIn("OPTIONAL_SETTING=\n", text)

    def test_keys_only_the_existing_file_has_are_kept(self) -> None:
        text, _, kept = merge(self.template, {"LOCAL_ONLY": "x"}, {})
        self.assertIn("LOCAL_ONLY=x", text)
        self.assertIn("LOCAL_ONLY", kept)


if __name__ == "__main__":
    unittest.main()
