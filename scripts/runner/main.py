"""CLI entry point: argv dispatch, and the one wiring of registry/renderer/runner."""

from __future__ import annotations

import sys
from pathlib import Path

from .config_loader import ConfigLoader
from .exceptions import RunnerError
from .execution import ScriptRunner
from .interactive import InteractiveShell
from .models import Lang
from .registry import ScriptRegistry
from .rendering import MenuRenderer

#: scripts/runner/main.py -> scripts/runner -> scripts -> repository root
ROOT_DIR = Path(__file__).resolve().parents[2]


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except AttributeError, ValueError:
        return False


class CliApp:
    """One fully wired orchestrator for a single process run. Constructing it only
    parses a few small JSON files, so tests can build one freely."""

    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self._root_dir = root_dir
        self.registry: ScriptRegistry = ConfigLoader(root_dir).load()
        self.renderer = MenuRenderer(self.registry, root_dir)
        self.runner = ScriptRunner(root_dir)
        self.shell = InteractiveShell(self.registry, self.renderer, self.runner)

    def parse_help_args(self, rest: list[str]) -> tuple[Lang, list[str]]:
        lang: Lang = self.registry.default_lang
        remaining: list[str] = []
        index = 0
        while index < len(rest):
            token = rest[index]
            if token == "--lang" and index + 1 < len(rest):
                lang = "ru" if rest[index + 1].lower().startswith("ru") else "en"
                index += 2
                continue
            remaining.append(token)
            index += 1
        return lang, remaining

    def run_help_command(self, rest: list[str]) -> int:
        lang, remaining = self.parse_help_args(rest)

        if not remaining:
            self.renderer.print_help_catalog(lang)
            return 0

        script = self.registry.find_script(remaining[0])
        if script is None:
            message = (
                f"Unknown script: {remaining[0]!r}"
                if lang == "en"
                else f"Неизвестный скрипт: {remaining[0]!r}"
            )
            print(message, file=sys.stderr)
            self.renderer.print_help_catalog(lang)
            return 2

        self.renderer.print_manual(script, lang)
        return 0

    def run_list_command(self) -> int:
        """Machine-readable catalog, one script per line.

        Exists for agents and shell scripts: parsing the decorated interactive menu
        would be guesswork, and an agent that has to guess eventually guesses wrong.
        """
        for script in self.registry.scripts_by_category_order():
            flags = []
            if script.destructive:
                flags.append("destructive")
            if script.platforms:
                flags.append("platforms=" + "|".join(script.platforms))
            print(
                f"{script.title}\t{script.category}\t{script.module}"
                f"\t{','.join(flags)}\t{script.summary.en}"
            )
        return 0

    def main(self, argv: list[str]) -> int:
        if argv and argv[0].lower() in ("help", "--help", "-h"):
            return self.run_help_command(argv[1:])
        if argv and argv[0].lower() in ("list", "--list"):
            return self.run_list_command()

        if not argv:
            if _stdin_is_interactive():
                return self.shell.run()
            # No arguments and no terminal — an agent, a CI step, or a pipe. Run the
            # documented default instead of blocking forever on input().
            return self.runner.run(self.registry.default_script(), [])

        script = self.registry.find_script(argv[0])
        if script is not None:
            return self.runner.run(script, argv[1:])

        # Leading flags with no script name are flags for the default script, so
        # `... --quick` keeps working without naming it. A bare word, though, was meant
        # as a script name: passing `gat` through would silently run the whole quality
        # gate instead of saying the name is wrong, and a typo must not be indistinguishable
        # from a deliberate choice.
        if not argv[0].startswith("-"):
            print(f"ERROR: unknown script: {argv[0]!r}", file=sys.stderr)
            print(
                "Run `list` for the catalog, or `help` for the manuals.",
                file=sys.stderr,
            )
            return 2
        return self.runner.run(self.registry.default_script(), argv)


def main(argv: list[str]) -> int:
    """The one function the root entry-point shim imports and calls."""
    try:
        return CliApp().main(argv)
    except RunnerError as error:
        # A broken hand-edit of the catalog should read as one clear line, not a
        # traceback through the loader.
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()
        return 130
    except EOFError:
        # Every menu prompt handles EOF itself and leaves cleanly; this catches any path
        # that does not, so the worst case is a quiet exit rather than a traceback.
        print()
        return 130
