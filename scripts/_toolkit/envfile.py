"""Reading and merging dotenv files without ever losing a value that is already set.

Shared infrastructure: dev-secrets writes .env with it, smoke reads credentials from it.

The format is the subset python-dotenv and Docker Compose both read: ``KEY=value`` lines,
``#`` comments, blank lines, and double-quoted values that may span lines — a PEM key is
exactly such a value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_NEEDS_QUOTES = set(" #'\n\t")


class EnvFileError(ValueError):
    """The file is not in the supported dotenv subset."""


@dataclass(frozen=True)
class Entry:
    """One logical line: an assignment, or a comment/blank line kept verbatim."""

    key: str | None
    value: str = ""
    raw: str = ""


def parse(text: str) -> list[Entry]:
    lines = text.replace("\r\n", "\n").split("\n")
    entries: list[Entry] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            entries.append(Entry(key=None, raw=line))
            index += 1
            continue
        match = _ASSIGNMENT.match(stripped)
        if match is None:
            raise EnvFileError(f"line {index + 1}: expected KEY=value")
        key, rest = match.groups()
        if rest.startswith('"'):
            body = rest[1:]
            parts: list[str] = []
            while not body.endswith('"'):
                parts.append(body)
                index += 1
                if index >= len(lines):
                    raise EnvFileError(f"{key}: quoted value is never closed")
                body = lines[index]
            parts.append(body[:-1])
            value = "\n".join(parts)
        else:
            value = rest.strip()
        entries.append(Entry(key=key, value=value))
        index += 1
    while entries and entries[-1].key is None and not entries[-1].raw.strip():
        entries.pop()
    return entries


def values(entries: list[Entry]) -> dict[str, str]:
    return {entry.key: entry.value for entry in entries if entry.key is not None}


def format_value(value: str) -> str:
    if '"' in value:
        raise EnvFileError("values containing a double quote are not supported")
    if value and _NEEDS_QUOTES.intersection(value):
        return f'"{value}"'
    return value


def merge(
    template: list[Entry],
    existing: Mapping[str, str],
    generators: Mapping[str, Callable[[], str]],
) -> tuple[str, list[str], list[str]]:
    """Render the template, filling each key from, in order: a non-empty existing value,
    the template's own default, a generator. Keys only the existing file has are kept
    at the end.

    Returns the rendered text, the keys that were generated, and the keys kept.
    """
    lines: list[str] = []
    generated: list[str] = []
    kept: list[str] = []
    template_keys: set[str] = set()

    for entry in template:
        if entry.key is None:
            lines.append(entry.raw)
            continue
        template_keys.add(entry.key)
        current = existing.get(entry.key, "")
        if current:
            value = current
            kept.append(entry.key)
        elif entry.value:
            value = entry.value
        elif entry.key in generators:
            value = generators[entry.key]()
            generated.append(entry.key)
        else:
            value = ""
        lines.append(f"{entry.key}={format_value(value)}")

    extras = [key for key in existing if key not in template_keys]
    if extras:
        lines.extend(["", "# Local additions not present in the template"])
        for key in extras:
            lines.append(f"{key}={format_value(existing[key])}")
            kept.append(key)

    return "\n".join(lines) + "\n", generated, kept
