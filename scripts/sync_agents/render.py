"""Assembling AGENTS.md from the rule modules in .ai/.

CLAUDE.md imports the modules natively, so only the Codex entry point is generated.
Codex reads at most 32 KiB of project instructions, so the assembled file has a budget:
a module marked ``<!-- tier: extended -->`` contributes only its title, path and
``> **Essence.**`` line instead of its full text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TIER_MARKER = "tier: extended"
ESSENCE_PREFIX = "> **Essence.**"

EXTENDED_INDEX_HEADER = (
    "<!-- module index: extended -->\n\n"
    "# Modules loaded on demand\n\n"
    "These rules bind exactly like the inlined ones; only their full text lives\n"
    "outside this file, to stay within the instruction budget. Read the file itself\n"
    "when a task touches it — that is an obligation, not a suggestion.\n"
)


class SyncError(RuntimeError):
    """The module tree is missing or empty."""


@dataclass(frozen=True)
class Module:
    relative_path: str
    body: str

    @property
    def is_extended(self) -> bool:
        return any(TIER_MARKER in line for line in self.body.splitlines()[:5])

    @property
    def title(self) -> str:
        for line in self.body.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return "(untitled module)"

    @property
    def essence(self) -> str | None:
        for line in self.body.splitlines():
            stripped = line.strip()
            if stripped.startswith(ESSENCE_PREFIX):
                return stripped[len(ESSENCE_PREFIX) :].strip()
        return None


def normalize(text: str) -> str:
    """Line endings are compared as LF, whatever the checkout converted them to."""
    return text.replace("\r\n", "\n")


def collect_modules(root: Path, groups: Sequence[str]) -> list[Module]:
    modules: list[Module] = []
    for group in groups:
        directory = root / ".ai" / group
        if not directory.is_dir():
            raise SyncError(f"missing module directory: {directory}")
        for path in sorted(directory.glob("*.md"), key=lambda item: item.name):
            body = normalize(path.read_text(encoding="utf-8")).strip()
            modules.append(Module(f".ai/{group}/{path.name}", body))
    if not modules:
        raise SyncError("no rule modules found under .ai/")
    return modules


def render(modules: Sequence[Module], banner: str) -> str:
    sections = [banner.rstrip()]
    extended = [module for module in modules if module.is_extended]

    for module in modules:
        if not module.is_extended:
            sections.append(f"<!-- module: {module.relative_path} -->\n\n{module.body}")

    if extended:
        index = EXTENDED_INDEX_HEADER
        for module in extended:
            index += f"\n## {module.title}\n\nFile: `{module.relative_path}`\n"
            if module.essence:
                index += f"\n{module.essence}\n"
        sections.append(index.rstrip())

    return "\n\n---\n\n".join(sections) + "\n"


def size_kib(content: str) -> float:
    return len(content.encode("utf-8")) / 1024
