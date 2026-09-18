"""Audit the locked dependencies for known vulnerabilities and write the reports.

Python: the exact versions of ``uv.lock`` (every workspace package, development tools
included) exported to a requirements file and checked by pip-audit against the PyPI and
OSV advisories. JavaScript: ``pnpm audit`` over the console's lock file. Both need the
internet, so this is not a gate section; CI runs it in its own job and keeps the reports.
The JSON reports land in ``audit-reports/``; the run fails when Python has any known
vulnerability or the console one at or above the configured severity.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts._toolkit.config import load_config, repo_root
from scripts._toolkit.console import fail, heading, info, ok, summary, warn
from scripts._toolkit.processes import capture_streams

SCRIPT_DIR = Path(__file__).resolve().parent
SEVERITIES = ("info", "low", "moderate", "high", "critical")


def python_findings(report: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(package, version, advisory, fixed in) for every known vulnerability."""
    findings = []
    for dependency in report.get("dependencies", []):
        for vulnerability in dependency.get("vulns", []):
            findings.append(
                (
                    str(dependency.get("name", "?")),
                    str(dependency.get("version", "?")),
                    str(vulnerability.get("id", "?")),
                    ", ".join(vulnerability.get("fix_versions", [])) or "no fix yet",
                )
            )
    return findings


def node_counts(report: dict[str, Any]) -> dict[str, int]:
    counts = report.get("metadata", {}).get("vulnerabilities", {})
    return {severity: int(counts.get(severity, 0)) for severity in SEVERITIES}


def fails_at(counts: dict[str, int], level: str) -> bool:
    threshold = SEVERITIES.index(level)
    return any(counts[severity] for severity in SEVERITIES[threshold:])


def audit_python(root: Path, config: dict[str, Any], reports: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as temp:
        requirements = Path(temp) / "requirements.txt"
        code, _, err = capture_streams(
            [
                "uv",
                "export",
                "--frozen",
                "--all-packages",
                "--no-emit-workspace",
                "--no-hashes",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements),
            ],
            root,
            timeout=120,
        )
        if code != 0:
            return False, f"uv export failed: {err.strip()[:200]}"
        code, out, err = capture_streams(
            [
                "uvx",
                "--from",
                f"pip-audit=={config['pip_audit_version']}",
                "pip-audit",
                "--requirement",
                str(requirements),
                "--no-deps",
                "--disable-pip",
                "--progress-spinner",
                "off",
                "--format",
                "json",
            ],
            root,
            timeout=600,
        )
    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return False, f"pip-audit gave no report (exit {code}): {err.strip()[:200]}"
    (reports / "pip-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    findings = python_findings(report)
    for package, version, advisory, fixed in findings:
        warn(f"{package} {version}: {advisory} (fixed in {fixed})")
    scanned = len(report.get("dependencies", []))
    if findings:
        return False, f"{len(findings)} known vulnerabilities in {scanned} packages"
    return True, f"{scanned} packages, no known vulnerabilities"


def audit_node(root: Path, config: dict[str, Any], reports: Path) -> tuple[bool, str]:
    web = root / str(config["web_dir"])
    if not (web / "node_modules").is_dir():
        return (
            False,
            "apps/web/node_modules is missing — run: pnpm --dir apps/web install",
        )
    code, out, err = capture_streams(
        ["pnpm", "--dir", str(web), "audit", "--json"], root, timeout=300
    )
    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return False, f"pnpm audit gave no report (exit {code}): {err.strip()[:200]}"
    (reports / "pnpm-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    counts = node_counts(report)
    listed = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if counts[s]) or "none"
    level = str(config["node_fail_level"])
    if fails_at(counts, level):
        return False, f"advisories: {listed} (fails at {level})"
    return True, f"advisories: {listed} (fails at {level})"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="audit-deps",
        description="Audit locked Python and console dependencies for known vulnerabilities.",
    )
    parser.add_argument("--report-dir", help="where to write the JSON reports")
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(SCRIPT_DIR, "audit.json")
    reports = root / str(args.report_dir or config["report_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    heading("audit-deps")
    rows: list[tuple[str, str]] = []
    passed = True
    for name, audit in (
        ("Python (uv.lock)", audit_python),
        ("console (pnpm)", audit_node),
    ):
        info(f"auditing {name}…")
        good, detail = audit(root, config, reports)
        (ok if good else fail)(f"{name}: {detail}")
        rows.append((name, "ok" if good else f"FAILED ({detail})"))
        passed = passed and good
    info(f"reports in {reports.relative_to(root)}")
    summary("audit-deps", rows)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
