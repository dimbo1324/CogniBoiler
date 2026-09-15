"""Loads and validates the JSON catalog that drives the orchestrator.

Every check here exists to turn a bad hand-edit into one immediate, specific error
instead of a cryptic failure deep inside rendering or — the case that actually
matters — a subprocess launching the wrong thing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import ConfigValidationError
from .models import MODULE_PATTERN, SUPPORTED_LANGS, Category, ScriptInfo, Text
from .registry import ScriptRegistry

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ConfigLoader:
    def __init__(self, root_dir: Path, config_dir: Path = CONFIG_DIR) -> None:
        self._root_dir = root_dir.resolve()
        self._config_dir = config_dir

    def load(self) -> ScriptRegistry:
        categories = self._load_categories()
        cadences = self._load_cadences()
        meta = self._load_meta()
        scripts = self._load_scripts(categories, cadences)
        return ScriptRegistry(
            categories=categories,
            scripts=scripts,
            default_script_title=meta["default_script_title"],
            default_lang=meta["default_lang"],
        )

    # -- file-level loaders -------------------------------------------------

    def _read_json(self, filename: str) -> Any:
        path = self._config_dir / filename
        if not path.exists():
            raise ConfigValidationError(f"missing config file: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"invalid JSON in {path}: {exc}") from exc

    def _load_categories(self) -> list[Category]:
        raw = self._read_json("categories.json")
        if not isinstance(raw, list):
            raise ConfigValidationError("categories.json must be a JSON array")

        categories: list[Category] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw):
            where = f"categories.json[{index}]"
            self._require_fields(entry, ("key", "title", "blurb"), where)
            key = entry["key"]
            if key in seen:
                raise ConfigValidationError(f"{where}: duplicate category key {key!r}")
            seen.add(key)
            categories.append(
                Category(
                    key=key,
                    title=self._text(entry["title"], f"{where}.title"),
                    blurb=self._text(entry["blurb"], f"{where}.blurb"),
                )
            )
        if not categories:
            raise ConfigValidationError("categories.json declares no categories")
        return categories

    def _load_cadences(self) -> dict[str, Text]:
        raw = self._read_json("cadences.json")
        if not isinstance(raw, dict):
            raise ConfigValidationError("cadences.json must be a JSON object")
        return {
            key: self._text(value, f"cadences.json[{key!r}]")
            for key, value in raw.items()
        }

    def _load_meta(self) -> dict[str, str]:
        raw = self._read_json("meta.json")
        if not isinstance(raw, dict):
            raise ConfigValidationError("meta.json must be a JSON object")
        self._require_fields(raw, ("default_script_title", "default_lang"), "meta.json")
        if raw["default_lang"] not in SUPPORTED_LANGS:
            raise ConfigValidationError(
                f"meta.json: default_lang {raw['default_lang']!r} is not supported "
                f"(known: {sorted(SUPPORTED_LANGS)})"
            )
        return raw

    def _load_scripts(
        self, categories: list[Category], cadences: dict[str, Text]
    ) -> list[ScriptInfo]:
        raw = self._read_json("scripts.json")
        if not isinstance(raw, list):
            raise ConfigValidationError("scripts.json must be a JSON array")

        category_keys = {c.key for c in categories}
        scripts: list[ScriptInfo] = []
        seen_identifiers: set[str] = set()

        for index, entry in enumerate(raw):
            where = f"scripts.json[{index}]"
            self._require_fields(
                entry,
                ("title", "module", "category", "summary", "description"),
                where,
            )

            category = entry["category"]
            if category not in category_keys:
                raise ConfigValidationError(
                    f"{where}: unknown category {category!r} "
                    f"(known: {sorted(category_keys)})"
                )

            module = entry["module"]
            self._require_scripts_module(module, where)

            script = ScriptInfo(
                title=entry["title"],
                module=module,
                category=category,
                summary=self._text(entry["summary"], f"{where}.summary"),
                description=self._text(entry["description"], f"{where}.description"),
                cadence=self._resolve_cadence(entry, cadences, where),
                aliases=tuple(entry.get("aliases", [])),
                examples=tuple(entry.get("examples", [])),
                platforms=tuple(entry.get("platforms", [])),
                destructive=bool(entry.get("destructive", False)),
            )

            for identifier in script.identifiers:
                if identifier in seen_identifiers:
                    raise ConfigValidationError(
                        f"{where}: duplicate identifier {identifier!r} — every "
                        "title and alias must be unique across the whole catalog"
                    )
                seen_identifiers.add(identifier)

            scripts.append(script)

        if not scripts:
            raise ConfigValidationError("scripts.json declares no scripts")
        return scripts

    # -- field-level helpers ------------------------------------------------

    def _require_fields(self, entry: Any, fields: tuple[str, ...], where: str) -> None:
        if not isinstance(entry, dict):
            raise ConfigValidationError(f"{where}: expected a JSON object")
        missing = [f for f in fields if f not in entry]
        if missing:
            raise ConfigValidationError(f"{where}: missing field(s) {missing}")

    def _text(self, raw: Any, where: str) -> Text:
        if not isinstance(raw, dict) or "en" not in raw or "ru" not in raw:
            raise ConfigValidationError(
                f"{where}: expected an object with 'en' and 'ru' string keys"
            )
        return Text(en=raw["en"], ru=raw["ru"])

    def _resolve_cadence(
        self, entry: dict[str, Any], cadences: dict[str, Text], where: str
    ) -> Text:
        has_inline = "cadence" in entry
        has_ref = "cadence_ref" in entry
        if has_inline and has_ref:
            raise ConfigValidationError(
                f"{where}: has both 'cadence' and 'cadence_ref' — pick one"
            )
        if has_ref:
            ref = entry["cadence_ref"]
            if ref not in cadences:
                raise ConfigValidationError(
                    f"{where}: cadence_ref {ref!r} not in cadences.json "
                    f"(known: {sorted(cadences)})"
                )
            return cadences[ref]
        if has_inline:
            return self._text(entry["cadence"], f"{where}.cadence")
        raise ConfigValidationError(
            f"{where}: must set either 'cadence' or 'cadence_ref'"
        )

    def _require_scripts_module(self, module: Any, where: str) -> None:
        """A script is launched with ``python -m <module>``. Anything importable would
        run, so the name is constrained to the ``scripts`` package by pattern — the
        module equivalent of refusing a path that escapes the project root. The
        directory must also exist, so a typo fails at load instead of at launch."""
        if not isinstance(module, str) or not MODULE_PATTERN.match(module):
            raise ConfigValidationError(
                f"{where}: module {module!r} must be a dotted name under 'scripts.', "
                "lowercase, e.g. 'scripts.clean_caches'"
            )
        package_dir = self._root_dir / Path(*module.split("."))
        if not package_dir.is_dir():
            raise ConfigValidationError(
                f"{where}: module {module!r} has no directory at {package_dir}"
            )
        if not (package_dir / "__main__.py").is_file():
            raise ConfigValidationError(
                f"{where}: {module!r} has no __main__.py, so `python -m {module}` "
                "would not run"
            )
