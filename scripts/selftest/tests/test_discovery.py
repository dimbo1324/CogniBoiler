"""Every test module under scripts/ is reachable by ``unittest discover``.

Discovery does not descend into a directory without ``__init__.py``, and it says nothing
about what it skipped: the dev-secrets certificate tests sat in such a directory and
never ran, while the step that should have run them stayed green.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]


def unreachable_test_modules(scripts: Path) -> list[str]:
    """Test modules below a directory that is not a package, as POSIX paths."""
    unreachable = []
    for module in sorted(scripts.rglob("test*.py")):
        chain = module.parent
        while chain != scripts and (chain / "__init__.py").is_file():
            chain = chain.parent
        if chain != scripts:
            unreachable.append(module.relative_to(scripts).as_posix())
    return unreachable


class DiscoveryTest(unittest.TestCase):
    def test_every_test_module_is_inside_a_package(self) -> None:
        self.assertEqual(unreachable_test_modules(SCRIPTS), [])

    def test_the_check_notices_a_directory_without_init(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            scripts = Path(temp)
            for package in ("tool", "tool/tests"):
                (scripts / package).mkdir()
                (scripts / package / "__init__.py").write_text("", encoding="utf-8")
            (scripts / "tool/tests/test_seen.py").write_text("", encoding="utf-8")
            (scripts / "other/tests").mkdir(parents=True)
            (scripts / "other/__init__.py").write_text("", encoding="utf-8")
            (scripts / "other/tests/test_hidden.py").write_text("", encoding="utf-8")
            self.assertEqual(
                unreachable_test_modules(scripts), ["other/tests/test_hidden.py"]
            )


if __name__ == "__main__":
    unittest.main()
