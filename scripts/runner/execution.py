"""Subprocess execution for catalog scripts."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .models import Lang, ScriptInfo


def split_arguments(raw: str) -> list[str]:
    """Split a typed argument string into argv, correctly on both platforms.

    ``shlex`` defaults to POSIX rules, where a backslash is an escape character. On
    Windows it is the path separator, so the default silently ate it: typing
    ``--out C:\\Users\\dev\\build`` produced ``['--out', 'C:Usersdevbuild']`` and the
    script received a path that does not exist. Nothing reported an error — the argument
    was simply wrong, which is the worst way for this to fail.

    Non-POSIX mode keeps backslashes but also keeps the quotes around a quoted token, so
    the surrounding pair is stripped afterwards to get what the shell would have passed.
    """
    if not raw.strip():
        return []
    if os.name != "nt":
        return shlex.split(raw)
    return [_unquote(token) for token in shlex.split(raw, posix=False)]


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


class ScriptRunner:
    """Launches a catalog script as ``python -m <module>``.

    The argv is always a list and ``shell`` is never true. This is the one place a
    string that came from config or from a user reaches a process launch, so it stays
    deliberately boring.

    Launching by module rather than by file path is what lets a one-file script and a
    fully decomposed one be invoked identically, and lets the decomposed one use
    ordinary relative imports without any ``sys.path`` manipulation.
    """

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir

    def run(self, script: ScriptInfo, extra_args: list[str]) -> int:
        completed = subprocess.run(
            [sys.executable, "-m", script.module, *extra_args],
            cwd=self._root_dir,
            check=False,
        )
        return completed.returncode

    def prompt_extra_args(self, script: ScriptInfo, lang: Lang) -> list[str]:
        prompt = (
            f"Extra arguments for {script.title} (optional, Enter to skip): "
            if lang == "en"
            else f"Дополнительные аргументы для {script.title} "
            "(необязательно, Enter — пропустить): "
        )
        try:
            raw = input(prompt).strip()
        except EOFError:
            # Ctrl+Z/Ctrl+D at an optional prompt reads as "skip", which is what Enter
            # does here anyway. Letting it propagate ended the session in a traceback.
            print()
            return []
        return split_arguments(raw)

    def launch(self, script: ScriptInfo, lang: Lang) -> int:
        return self.run(script, self.prompt_extra_args(script, lang))
