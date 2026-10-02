"""install-hooks installs pre-commit from the project environment.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest
from unittest import mock

from scripts.install_hooks import __main__ as install_hooks


class InstallHooksTest(unittest.TestCase):
    def test_the_hook_comes_from_the_locked_environment(self) -> None:
        with mock.patch.object(install_hooks, "run_steps", return_value=0) as run_steps:
            self.assertEqual(install_hooks.main([]), 0)
        (step,) = run_steps.call_args.args[0]
        self.assertEqual(step["argv"][:3], ["uv", "run", "--no-sync"])
        self.assertEqual(
            step["argv"][-3:], ["pre-commit", "install", "--install-hooks"]
        )


if __name__ == "__main__":
    unittest.main()
