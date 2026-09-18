"""How audit-deps reads the pip-audit and pnpm audit reports.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import unittest

from scripts.audit_deps.__main__ import fails_at, node_counts, python_findings


class PythonReportTest(unittest.TestCase):
    def test_every_advisory_of_every_package_is_listed_with_its_fix(self) -> None:
        report = {
            "dependencies": [
                {
                    "name": "starlette",
                    "version": "0.52.1",
                    "vulns": [
                        {"id": "PYSEC-1", "fix_versions": ["1.0.1"]},
                        {"id": "PYSEC-2", "fix_versions": []},
                    ],
                },
                {"name": "anyio", "version": "4.14.2", "vulns": []},
            ]
        }
        self.assertEqual(
            python_findings(report),
            [
                ("starlette", "0.52.1", "PYSEC-1", "1.0.1"),
                ("starlette", "0.52.1", "PYSEC-2", "no fix yet"),
            ],
        )


class NodeReportTest(unittest.TestCase):
    def test_counts_default_to_zero(self) -> None:
        counts = node_counts({"metadata": {"vulnerabilities": {"high": 2}}})
        self.assertEqual(counts["high"], 2)
        self.assertEqual(counts["critical"], 0)

    def test_the_threshold_includes_everything_more_severe(self) -> None:
        counts = node_counts({"metadata": {"vulnerabilities": {"critical": 1}}})
        self.assertTrue(fails_at(counts, "high"))
        moderate = node_counts({"metadata": {"vulnerabilities": {"moderate": 3}}})
        self.assertFalse(fails_at(moderate, "high"))
        self.assertTrue(fails_at(moderate, "moderate"))


if __name__ == "__main__":
    unittest.main()
