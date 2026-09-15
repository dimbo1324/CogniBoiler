"""Deciding what clean-caches may delete. Walks the tree; never deletes anything.

Protection is by name and by link: a protected directory is not descended into, and a
symlink or junction is never followed or selected, so nothing outside the repository
and nothing inside .venv, node_modules or .git can ever reach the deletion list.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rules:
    directory_names: frozenset[str]
    file_names: frozenset[str]
    extra_paths: tuple[str, ...]
    protected_names: frozenset[str]

    @classmethod
    def from_config(cls, config: dict[str, Iterable[str]]) -> Rules:
        return cls(
            directory_names=frozenset(config["directory_names"]),
            file_names=frozenset(config["file_names"]),
            extra_paths=tuple(config["extra_paths"]),
            protected_names=frozenset(config["protected_names"]),
        )


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root)


def plan(root: Path, rules: Rules) -> list[Path]:
    root = root.resolve()
    targets: set[Path] = set()

    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        here = Path(current)
        descend: list[str] = []
        for name in sorted(dirnames):
            path = here / name
            if name in rules.protected_names or _is_link(path):
                continue
            if name in rules.directory_names:
                targets.add(path)
                continue
            descend.append(name)
        dirnames[:] = descend

        for name in filenames:
            path = here / name
            if name in rules.file_names and not _is_link(path):
                targets.add(path)

    for relative in rules.extra_paths:
        path = root / relative
        protected = rules.protected_names.intersection(Path(relative).parts)
        if (
            path.exists()
            and not protected
            and not _is_link(path)
            and _inside(root, path)
        ):
            targets.add(path)

    return sorted(targets)
