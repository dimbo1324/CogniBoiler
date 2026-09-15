"""The interactive shell: top menu → category → launch, plus the help browser.

Every ``input`` call in the orchestrator lives here.
"""

from __future__ import annotations

from .execution import ScriptRunner
from .models import Category, ScriptInfo, Session
from .registry import ScriptRegistry
from .rendering import MenuRenderer, platform_note, unknown_choice_message

_BACK_WORDS = ("b", "back", "назад")
_QUIT_WORDS = ("q", "quit", "exit", "выход")
_LANG_WORDS = ("l", "lang", "language", "язык")
_HELP_WORDS = ("h", "help", "справка")

#: Returned by ``_ask`` when the input stream ends. A distinct sentinel rather than an
#: empty string, because empty means "the user pressed Enter" and that is a real choice
#: here — it runs the default script.
_EOF = object()


def _ask(prompt: str) -> str | object:
    """Read one answer, or ``_EOF`` if the stream ended.

    Ctrl+Z on Windows and Ctrl+D elsewhere are ordinary ways to leave a menu, and they
    used to end the session with an ``EOFError`` traceback across the terminal.
    """
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return _EOF


class InteractiveShell:
    def __init__(
        self,
        registry: ScriptRegistry,
        renderer: MenuRenderer,
        runner: ScriptRunner,
    ) -> None:
        self._registry = registry
        self._renderer = renderer
        self._runner = runner

    def run(self) -> int:
        session = Session(lang=self._registry.default_lang)
        while True:
            self._renderer.print_top_menu(session.lang)
            answer = _ask(self._renderer.top_prompt(session.lang))
            if answer is _EOF:
                return 0
            raw = str(answer)
            lowered = raw.lower()

            if not raw:
                return self._launch(self._registry.default_script(), session)
            if lowered in _QUIT_WORDS:
                return 0
            if lowered in _HELP_WORDS:
                self._run_help_browser(session)
                continue
            if lowered in _LANG_WORDS:
                session.toggle_lang()
                continue

            script = self._registry.find_script(raw)
            if script is not None:
                return self._launch(script, session)

            categories = self._registry.categories
            category = (
                categories[int(raw) - 1]
                if raw.isdigit() and 1 <= int(raw) <= len(categories)
                else self._registry.find_category(raw)
            )
            if category is not None:
                outcome = self._run_category(category, session)
                if outcome is not None:
                    return outcome
                continue

            print(unknown_choice_message(raw, session.lang))

    def _launch(self, script: ScriptInfo, session: Session) -> int:
        """Print the platform note, if any, before handing over.

        Deliberately not a refusal — see ``rendering.platform_note``.
        """
        note = platform_note(script, session.lang)
        if note:
            print(f"\n{note}")
        return self._runner.launch(script, session.lang)

    def _run_help_browser(self, session: Session) -> None:
        # scripts_by_category_order(), not registry.scripts: this has to match what
        # print_help_catalog numbered on screen, or a typed digit opens the wrong
        # manual.
        catalog_order = self._registry.scripts_by_category_order()
        while True:
            self._renderer.print_help_catalog(session.lang)
            answer = _ask(self._renderer.help_prompt(session.lang))
            if answer is _EOF:
                return
            raw = str(answer)
            lowered = raw.lower()

            if not raw or lowered in _BACK_WORDS:
                return
            if lowered in _LANG_WORDS:
                session.toggle_lang()
                continue

            script: ScriptInfo | None = None
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(catalog_order):
                    script = catalog_order[index - 1]
            if script is None:
                script = self._registry.find_script(raw)

            if script is None:
                print(unknown_choice_message(raw, session.lang))
                continue
            self._renderer.print_manual(script, session.lang)

    def _run_category(self, category: Category, session: Session) -> int | None:
        """Returns an exit code once something ran or the user quit, or ``None`` to
        go back to the top menu."""
        scripts = self._registry.scripts_in(category.key)
        while True:
            self._renderer.print_category_menu(category, scripts, session.lang)
            answer = _ask(self._renderer.category_prompt(session.lang))
            if answer is _EOF:
                return 0
            raw = str(answer)
            lowered = raw.lower()

            if not raw or lowered in _BACK_WORDS:
                return None
            if lowered in _QUIT_WORDS:
                return 0
            if lowered in _LANG_WORDS:
                session.toggle_lang()
                continue
            if lowered in _HELP_WORDS:
                for entry in scripts:
                    self._renderer.print_manual(entry, session.lang)
                continue

            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(scripts):
                    return self._launch(scripts[index - 1], session)
                print(unknown_choice_message(raw, session.lang))
                continue

            script = self._registry.find_script(raw)
            if script is not None:
                return self._launch(script, session)

            print(unknown_choice_message(raw, session.lang))
