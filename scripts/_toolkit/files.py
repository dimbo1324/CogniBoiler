"""Writing files that hold secrets: whole, and readable by their owner only.

``Path.write_text`` creates a file with the process umask, typically 0644 on POSIX, and
replacing a file through a temporary swaps in that new inode — so a developer who ran
``chmod 600 .env`` got 0644 back the next time the file was regenerated. On Windows the
file inherits the folder's ACL, which is already private to the user, so plain writes
are kept there.
"""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_FILE = 0o600
PRIVATE_DIR = 0o700


def write_private(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` (UTF-8, LF), atomically, mode 0600 on POSIX."""
    temporary = path.with_name(path.name + ".tmp")
    if os.name == "nt":
        temporary.write_text(text, encoding="utf-8", newline="\n")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        descriptor = os.open(temporary, flags, PRIVATE_FILE)
        # O_CREAT's mode applies only to a new file; a temporary left behind by an
        # interrupted run keeps whatever mode it had, so it is set explicitly.
        os.fchmod(descriptor, PRIVATE_FILE)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    os.replace(temporary, path)


def make_private_dir(path: Path) -> None:
    """Create a new directory, and its parents, that only its owner may enter on POSIX.

    Raises ``FileExistsError`` when it already exists: a private directory is always a
    fresh one, never an existing folder whose contents someone else may already read.
    """
    path.mkdir(parents=True, exist_ok=False)
    if os.name == "posix":
        path.chmod(PRIVATE_DIR)
