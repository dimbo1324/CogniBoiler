"""A queryable, read-only view over the loaded catalog. Built once, never mutated."""

from __future__ import annotations

from .exceptions import RunnerError
from .models import Category, Lang, ScriptInfo


class ScriptRegistry:
    def __init__(
        self,
        categories: list[Category],
        scripts: list[ScriptInfo],
        default_script_title: str,
        default_lang: Lang,
    ) -> None:
        self._categories = categories
        self._scripts = scripts
        self._default_script_title = default_script_title
        self._default_lang = default_lang

    @property
    def default_lang(self) -> Lang:
        """The interface language from `meta.json`.

        Read from config rather than fixed in Python so that changing the JSON actually
        changes behaviour — the key was previously required by the loader and read by
        nobody, which is worse than not having it.
        """
        return self._default_lang

    @property
    def categories(self) -> list[Category]:
        return list(self._categories)

    @property
    def scripts(self) -> list[ScriptInfo]:
        return list(self._scripts)

    def find_script(self, choice: str) -> ScriptInfo | None:
        normalized = choice.strip().lower()
        for script in self._scripts:
            if normalized in script.identifiers:
                return script
        return None

    def find_category(self, choice: str) -> Category | None:
        normalized = choice.strip().lower()
        for category in self._categories:
            names = {
                category.key,
                category.title.en.lower(),
                category.title.ru.lower(),
            }
            if normalized in names:
                return category
        return None

    def scripts_in(self, category_key: str) -> list[ScriptInfo]:
        return [s for s in self._scripts if s.category == category_key]

    def scripts_by_category_order(self) -> list[ScriptInfo]:
        """Every script, grouped by category in category-declaration order.

        This is the single source of truth for "the Nth entry", so a digit typed in
        the help browser always opens the entry that was printed as N. Using raw
        declaration order instead would drift the moment a script is declared away
        from the rest of its category.
        """
        ordered: list[ScriptInfo] = []
        for category in self._categories:
            ordered.extend(self.scripts_in(category.key))
        return ordered

    def default_script(self) -> ScriptInfo:
        script = self.find_script(self._default_script_title)
        if script is None:
            raise RunnerError(
                f"default_script_title {self._default_script_title!r} in meta.json "
                "matches no registered script"
            )
        return script

    def category_for(self, script: ScriptInfo) -> Category:
        for category in self._categories:
            if category.key == script.category:
                return category
        # The loader proves every reference resolves, so reaching here means a
        # registry was assembled by hand (a test) with inconsistent data.
        raise RunnerError(
            f"script {script.title!r} references unknown category {script.category!r}"
        )
