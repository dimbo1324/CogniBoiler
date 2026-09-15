"""Immutable value objects for the orchestrator.

Pure data: no I/O, no knowledge of where config comes from. ``ConfigLoader`` is the
only place that turns raw JSON into these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

Lang = str  # "en" or "ru"

#: Every language the UI copy is written in. The *default* is not here — it comes from
#: `config/meta.json`, because a hardcoded default is a setting the owner cannot change
#: by editing the JSON, and this catalog exists so that settings live in JSON.
SUPPORTED_LANGS: frozenset[Lang] = frozenset({"en", "ru"})

#: Used only where no loaded config is available — a failure so early that `meta.json`
#: could not be read, and the error message still has to come out in some language.
FALLBACK_LANG: Lang = "en"

#: A script is launched as ``python -m <module>``, and the module must live under the
#: ``scripts`` package. Validating the shape here is what stops a hand-edited catalog
#: from pointing the launcher at an arbitrary importable module.
MODULE_PATTERN = re.compile(r"^scripts\.[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")


@dataclass(frozen=True)
class Text:
    """One piece of UI copy in both supported languages.

    English is the default because tooling and agent-facing output is English by the
    project's own language policy; Russian is always one keystroke away because the
    owner reads reports in Russian.
    """

    en: str
    ru: str

    def get(self, lang: Lang) -> str:
        return self.ru if lang == "ru" else self.en


@dataclass(frozen=True)
class Category:
    """A menu grouping. Scripts reference one by ``key``; the loader proves every
    reference resolves before the menu can render."""

    key: str
    title: Text
    blurb: Text


@dataclass(frozen=True)
class ScriptInfo:
    """One entry in the catalog.

    ``platforms`` is the set of ``sys.platform`` prefixes a script is *useful* on —
    empty means "anywhere". It never blocks a launch: the orchestrator is
    cross-platform on purpose, so it prints the mismatch and lets the script itself
    give the precise refusal, which is the only place that knows why.
    """

    title: str
    module: str
    category: str
    summary: Text
    description: Text
    cadence: Text
    aliases: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)
    platforms: tuple[str, ...] = field(default_factory=tuple)
    destructive: bool = False

    @property
    def identifiers(self) -> frozenset[str]:
        """Every string a user could type to pick this script — case-folded, since
        lookups normalise to lowercase before matching."""
        return frozenset({self.title.lower(), *(a.lower() for a in self.aliases)})

    @property
    def package_dir(self) -> Path:
        """Where this script's own files live, relative to the repository root."""
        return Path(*self.module.split("."))


@dataclass
class Session:
    """Mutable per-run interactive state — currently just the display language,
    threaded through every menu so a switch anywhere sticks for the rest of the run."""

    lang: Lang = FALLBACK_LANG

    def toggle_lang(self) -> None:
        self.lang = "ru" if self.lang == "en" else "en"
