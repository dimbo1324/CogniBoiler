"""Making console output survive whatever encoding the terminal happens to use.

Every script in this repository printed a heading through ``console.heading`` before it
did any work, and every one of those headings contains an em dash. On the default code
page of a Russian ``cmd.exe`` (cp866) that single character raises ``UnicodeEncodeError``
and the script dies before reaching its first real step — a tool that cannot run at all
on the platform the product targets.

Cyrillic itself was never the problem: cp866 and cp1251 both encode it. The whole crash
came from two typographic characters the copy happens to use, ``—`` and ``→``.

Two rules, and the split between them is the point:

* **A terminal keeps its own encoding**, and unencodable characters degrade to an ASCII
  equivalent. Forcing UTF-8 bytes at a cp866 console would replace a crash with mojibake,
  which is not obviously better — whereas ``--`` instead of ``—`` costs a reader nothing.
* **A redirected stream is switched to UTF-8.** A file or a pipe is a byte sink with no
  rendering opinion, so the typographic character survives, and a CI job whose locale is
  ASCII gets readable Russian instead of a page of question marks.

The console code page is deliberately **not** modified. ``SetConsoleOutputCP`` outlives
the process and would leave the user's terminal changed after a script they ran once.
"""

from __future__ import annotations

import codecs
import sys
from typing import IO, Any

#: ASCII stand-ins for the non-ASCII punctuation this repository's script output uses.
#: Only characters that actually appear are listed: an exhaustive transliteration table
#: would be a library, and the fallback below already handles everything else.
_SUBSTITUTIONS = {
    "—": "--",  # em dash
    "–": "-",  # en dash
    "→": "->",  # rightwards arrow
    "←": "<-",  # leftwards arrow
    "…": "...",  # horizontal ellipsis
    " ": " ",  # no-break space
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "•": "*",  # bullet
    "✓": "ok",  # check mark
    "✗": "x",  # ballot x
}

#: The name the error handler is registered under. Registration is global to the
#: interpreter and idempotent, so a second call is harmless.
ERROR_HANDLER = "cogniboiler.degrade"

_registered = False


def _degrade(error: UnicodeError) -> tuple[str, int]:
    """Replace characters the target encoding cannot represent.

    Returns a replacement plus the index to resume at, which is the contract
    ``codecs.register_error`` expects. Anything without a listed substitute becomes
    ``?`` rather than raising: this handler exists so that output is never the thing
    that fails, and a script that dies while reporting a result is strictly worse than
    one whose dash came out as two hyphens.
    """
    if not isinstance(error, UnicodeEncodeError):
        raise error
    chunk = error.object[error.start : error.end]
    replacement = "".join(_SUBSTITUTIONS.get(character, "?") for character in chunk)
    return replacement, error.end


def _register() -> None:
    global _registered
    if _registered:
        return
    codecs.register_error(ERROR_HANDLER, _degrade)
    _registered = True


def _harden(stream: IO[Any] | None) -> None:
    """Apply the two rules to one stream, tolerating every stream that cannot comply.

    ``reconfigure`` exists only on ``TextIOWrapper``; under pytest capture, in an
    embedded interpreter, or behind a custom wrapper the stream may be something else
    entirely. A diagnostic convenience must never be the reason a script fails, so every
    failure here is swallowed and the stream is left exactly as it was.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    try:
        is_terminal = bool(stream.isatty())  # type: ignore[union-attr]
    except AttributeError, ValueError:
        is_terminal = False

    try:
        if is_terminal:
            reconfigure(errors=ERROR_HANDLER)
        else:
            reconfigure(encoding="utf-8", errors=ERROR_HANDLER)
    except ValueError, OSError, LookupError:
        # A closed or detached stream, or an encoding this build has no codec for.
        return


def enable_safe_output() -> None:
    """Make ``sys.stdout``/``sys.stderr`` unable to raise on an unencodable character.

    Called from ``scripts/__init__.py`` so it takes effect before any script's first
    print, including one from a module-level failure. Idempotent.
    """
    _register()
    _harden(sys.stdout)
    _harden(sys.stderr)


def ascii_fallback(text: str) -> str:
    """``text`` with the known non-ASCII punctuation replaced, for callers that need a
    plain-ASCII string rather than a stream that degrades on write."""
    for character, replacement in _SUBSTITUTIONS.items():
        text = text.replace(character, replacement)
    return text
