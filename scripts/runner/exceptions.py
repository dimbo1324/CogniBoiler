"""Exception types for the dev-tools orchestrator."""

from __future__ import annotations


class RunnerError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigValidationError(RunnerError):
    """Raised when a catalog JSON file is missing, malformed, or internally
    inconsistent: an unknown category reference, a duplicate identifier, a module
    that does not live under ``scripts.``.

    The catalog is hand-edited — that is the entire reason it is JSON rather than
    Python — so a bad edit has to produce one clear message naming the file and the
    entry. It must never surface as a bare ``KeyError`` from inside menu rendering,
    and above all never as a subprocess launching something other than what the
    author meant.
    """
