"""What dev-secrets writes to disk: a complete .env that only its owner can read.

Needs the cryptography package, like the script itself; the gate runs these tests in the
project environment, where it is installed.

Run with:  uv run --no-sync python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.dev_secrets import __main__ as dev_secrets

HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None

TEMPLATE = (
    "# local development\n"
    "POSTGRES_USER=cogniboiler\n"
    "POSTGRES_PASSWORD=\n"
    "JWT_PRIVATE_KEY=\n"
    "JWT_PUBLIC_KEY=\n"
)


class EnvironmentTest(unittest.TestCase):
    def test_ci_cannot_skip_the_key_material_tests(self) -> None:
        """Locally a bare interpreter may lack cryptography and skip them; under CI the
        gate runs them in the project environment, so a skip there hides a defect."""
        if os.environ.get("CI"):
            self.assertTrue(
                HAS_CRYPTOGRAPHY,
                "cryptography is missing: run the scripts' tests through uv run",
            )


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "cryptography is not installed")
class WriteEnvTest(unittest.TestCase):
    def _run(self, root: Path) -> int:
        with (
            mock.patch.object(dev_secrets, "repo_root", return_value=root),
            mock.patch.object(dev_secrets, "capture", return_value=(0, "")),
            contextlib.redirect_stdout(io.StringIO()) as printed,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = dev_secrets.main([])
        self.printed = printed.getvalue()
        return code

    def test_missing_secrets_are_generated_and_set_values_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env.example").write_text(TEMPLATE, encoding="utf-8")
            (root / ".env").write_text("POSTGRES_PASSWORD=kept\n", encoding="utf-8")
            self.assertEqual(self._run(root), 0)
            written = (root / ".env").read_text(encoding="utf-8")
        self.assertIn("POSTGRES_PASSWORD=kept\n", written)
        self.assertIn("-----BEGIN PRIVATE KEY-----", written)
        self.assertIn("generated JWT_PRIVATE_KEY", self.printed)
        self.assertNotIn("BEGIN", self.printed)

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX permissions")
    def test_the_env_file_is_readable_by_its_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env.example").write_text(TEMPLATE, encoding="utf-8")
            (root / ".env").write_text("POSTGRES_PASSWORD=kept\n", encoding="utf-8")
            (root / ".env").chmod(0o644)
            self.assertEqual(self._run(root), 0)
            mode = stat.S_IMODE((root / ".env").stat().st_mode)
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
