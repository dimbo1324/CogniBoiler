"""Output must never be the reason a script fails.

Every script printed a heading containing an em dash before doing any work. On the
default code page of a Russian ``cmd.exe`` (cp866) that raised ``UnicodeEncodeError``
and killed the process — on the one platform the product targets.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import io
import unittest

from scripts._toolkit import terminal


class DegradingErrorHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        terminal.enable_safe_output()

    def _write(self, text: str, encoding: str) -> str:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(
            buffer, encoding=encoding, errors=terminal.ERROR_HANDLER
        )
        stream.write(text)
        stream.flush()
        return buffer.getvalue().decode(encoding)

    def test_an_em_dash_does_not_kill_a_cp866_console(self) -> None:
        # The exact string shape every script's first heading has.
        self.assertEqual(
            self._write("cogniboiler — check", "cp866"), "cogniboiler -- check"
        )

    def test_cyrillic_survives_the_legacy_russian_code_pages(self) -> None:
        # Cyrillic was never the problem, and a fix that mangled it to fix the dash
        # would have broken the Russian menus this project ships on purpose.
        for encoding in ("cp866", "cp1251"):
            with self.subTest(encoding=encoding):
                self.assertEqual(self._write("Качество", encoding), "Качество")

    def test_an_arrow_degrades_rather_than_raising(self) -> None:
        self.assertEqual(self._write("a → b", "cp1251"), "a -> b")

    def test_an_unlisted_character_becomes_a_question_mark(self) -> None:
        # No substitution table can be complete. The contract is "never raise", not
        # "always transliterate".
        self.assertEqual(self._write("emoji \U0001f600", "cp1251"), "emoji ?")

    def test_utf8_is_untouched(self) -> None:
        self.assertEqual(self._write("cogniboiler — ✓", "utf-8"), "cogniboiler — ✓")


class AsciiFallbackTest(unittest.TestCase):
    def test_it_replaces_the_known_punctuation(self) -> None:
        self.assertEqual(terminal.ascii_fallback("a — b → c…"), "a -- b -> c...")


class HardenTest(unittest.TestCase):
    def test_a_stream_that_cannot_reconfigure_is_left_alone(self) -> None:
        # Under test capture, in an embedded interpreter, or behind a custom wrapper,
        # stdout may not be a TextIOWrapper at all. A diagnostic convenience must not be
        # the thing that breaks a script.
        class Bare:
            pass

        terminal._harden(Bare())  # must not raise
        terminal._harden(None)

    def test_a_redirected_stream_is_switched_to_utf8(self) -> None:
        # A file or pipe has no rendering opinion, so the typographic character should
        # survive rather than degrade — and an ASCII locale in CI should not turn
        # Russian output into a page of question marks.
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="ascii")
        terminal._harden(stream)
        self.assertEqual(stream.encoding.lower().replace("-", ""), "utf8")
        stream.write("Качество —")
        stream.flush()
        self.assertEqual(buffer.getvalue().decode("utf-8"), "Качество —")


if __name__ == "__main__":
    unittest.main()
