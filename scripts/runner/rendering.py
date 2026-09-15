"""Every ``print`` the orchestrator does.

This module never reads input and ``interactive.py`` never formats a menu, so both
stay testable on their own.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .models import Category, Lang, ScriptInfo
from .registry import ScriptRegistry

_RULE = "=" * 78
_THIN = "-" * 78


def unknown_choice_message(choice: str, lang: Lang) -> str:
    return (
        f"Unknown choice: {choice!r}. Type 'h' for help or 'q' to quit."
        if lang == "en"
        else f"Неизвестный выбор: {choice!r}. 'h' — справка, 'q' — выход."
    )


def platform_note(script: ScriptInfo, lang: Lang) -> str | None:
    """Warn when a script is not meant for this host, and return ``None`` otherwise.

    A warning rather than a refusal: the orchestrator runs everywhere by design, and
    the script itself is the only thing that knows precisely why it cannot proceed
    and what to do instead. Blocking here would replace a good message with a vague
    one.
    """
    if not script.platforms or sys.platform.startswith(script.platforms):
        return None
    supported = ", ".join(script.platforms)
    return (
        f"NOTE: {script.title} targets {supported}; this host is {sys.platform}."
        if lang == "en"
        else f"ВНИМАНИЕ: {script.title} предназначен для {supported}; "
        f"текущая система — {sys.platform}."
    )


class MenuRenderer:
    def __init__(self, registry: ScriptRegistry, root_dir: Path) -> None:
        self._registry = registry
        self._root_dir = root_dir

    # -- prompts ------------------------------------------------------------

    def top_prompt(self, lang: Lang) -> str:
        return (
            "Category number, script name, or Enter for the default: "
            if lang == "en"
            else "Номер категории, имя скрипта или Enter — по умолчанию: "
        )

    def category_prompt(self, lang: Lang) -> str:
        return (
            "Script number or name ('b' back, 'q' quit): "
            if lang == "en"
            else "Номер или имя скрипта ('b' назад, 'q' выход): "
        )

    def help_prompt(self, lang: Lang) -> str:
        return (
            "Entry number or script name ('b' back): "
            if lang == "en"
            else "Номер записи или имя скрипта ('b' назад): "
        )

    # -- menus --------------------------------------------------------------

    def print_top_menu(self, lang: Lang) -> None:
        default = self._registry.default_script()
        print(f"\n{_RULE}")
        print(
            "CogniBoiler dev tools"
            if lang == "en"
            else "CogniBoiler: инструменты разработки"
        )
        print(_RULE)
        print(f"  {self._root_dir}")
        print(f"  python {sys.version.split()[0]} on {sys.platform}")
        print(_THIN)

        for index, category in enumerate(self._registry.categories, start=1):
            count = len(self._registry.scripts_in(category.key))
            print(f"  {index}. {category.title.get(lang)}  ({count})")
            print(f"       {category.blurb.get(lang)}")

        print(_THIN)
        if lang == "en":
            print(f"  Enter — run the default: {default.title}")
            print("  h — help catalog     l — language (RU/EN)     q — quit")
        else:
            print(f"  Enter — запустить по умолчанию: {default.title}")
            print("  h — каталог справки  l — язык (RU/EN)         q — выход")
        print(_RULE)

    def print_category_menu(
        self, category: Category, scripts: list[ScriptInfo], lang: Lang
    ) -> None:
        print(f"\n{_RULE}")
        print(category.title.get(lang))
        print(_RULE)
        print(f"  {category.blurb.get(lang)}")
        print(_THIN)
        for index, script in enumerate(scripts, start=1):
            marks = []
            if script.destructive:
                marks.append("DELETES FILES" if lang == "en" else "УДАЛЯЕТ ФАЙЛЫ")
            if platform_note(script, lang):
                marks.append(sys.platform + "?" if lang == "en" else "не для этой ОС")
            suffix = f"   [{', '.join(marks)}]" if marks else ""
            print(f"  {index}. {script.title}{suffix}")
            print(f"       {script.summary.get(lang)}")
        print(_THIN)
        print(
            "  b — back     h — show manuals     l — language     q — quit"
            if lang == "en"
            else "  b — назад    h — руководства      l — язык         q — выход"
        )
        print(_RULE)

    # -- help ---------------------------------------------------------------

    def print_help_catalog(self, lang: Lang) -> None:
        print(f"\n{_RULE}")
        print("Script catalog" if lang == "en" else "Каталог скриптов")
        print(_RULE)
        number = 0
        for category in self._registry.categories:
            scripts = self._registry.scripts_in(category.key)
            if not scripts:
                continue
            print(f"\n{category.title.get(lang)}")
            for script in scripts:
                number += 1
                print(f"  {number:>2}. {script.title}")
                print(f"      {script.summary.get(lang)}")
        print(f"\n{_THIN}")
        print(
            "  Pick a number for the full manual. 'b' — back."
            if lang == "en"
            else "  Номер — полное руководство. 'b' — назад."
        )
        print(_RULE)

    def print_manual(self, script: ScriptInfo, lang: Lang) -> None:
        category = self._registry.category_for(script)
        print(f"\n{_RULE}")
        print(script.title)
        print(_RULE)
        print(
            f"  Category:  {category.title.get(lang)}"
            if lang == "en"
            else f"  Категория: {category.title.get(lang)}"
        )
        print(f"  Module:    python -m {script.module}")
        print(f"  Files:     {script.package_dir}/")
        if script.platforms:
            print(f"  Platforms: {', '.join(script.platforms)}")
        if script.destructive:
            print(
                "  WARNING:   deletes files — read its own --help before running"
                if lang == "en"
                else "  ВНИМАНИЕ:  удаляет файлы — прочитайте его --help перед запуском"
            )
        print(_THIN)
        print(f"  {script.description.get(lang)}")
        print(_THIN)
        print(
            f"  When to run: {script.cadence.get(lang)}"
            if lang == "en"
            else f"  Когда запускать: {script.cadence.get(lang)}"
        )
        if script.examples:
            print(_THIN)
            print("  Examples:" if lang == "en" else "  Примеры:")
            for example in script.examples:
                print(
                    f"    python dev_tools_scripts_runner.py {script.title} {example}"
                )
        note = platform_note(script, lang)
        if note:
            print(_THIN)
            print(f"  {note}")
        print(_RULE)
