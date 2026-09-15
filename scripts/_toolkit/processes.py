"""Launching external tools, portably.

The whole reason this module exists is that "run a command" is not portable, and every
script needs it. Getting it wrong once, here, is far better than getting it wrong seven
times in seven scripts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Tools that are not a plain executable on Windows. npm-installed CLIs land as
#: ``<name>.CMD`` shims, and ``CreateProcess`` only ever appends ``.exe`` when resolving
#: a bare name — so a bare "pnpm" is reported as missing on a machine that has it.
#:
#: The bare-name case is handled by the ``shutil.which`` call that runs first, so it is
#: not repeated here — an empty entry in this tuple was skipped by the loop and only
#: looked like coverage it did not provide.
_WINDOWS_SHIM_SUFFIXES = (".cmd", ".bat", ".exe")

#: Seconds a *captured* command may take before it is abandoned. Capturing means the
#: caller is waiting on output it intends to parse, so a tool that never answers hangs
#: the script with no sign of why. The classic on Windows is the Store's ``python``
#: stub, which opens a shop window and never returns.
#:
#: Streaming commands (``run``) deliberately have no timeout: those are builds and test
#: suites, where "slow" is normal and the user can see progress and interrupt.
DEFAULT_CAPTURE_TIMEOUT = 30.0

#: Synthetic exit codes, following the shell conventions so they are not mistaken for a
#: tool's own. 127 is "command not found" everywhere; 124 is what GNU ``timeout`` returns.
NOT_FOUND = 127
TIMED_OUT = 124


class ToolNotFoundError(RuntimeError):
    """A required external tool is not on PATH."""


def find_tool(name: str) -> str | None:
    """Absolute path to ``name``, or ``None``.

    On Windows this tries the shim suffixes explicitly rather than trusting a bare
    lookup, because that is where npm-installed CLIs live.
    """
    direct = shutil.which(name)
    if direct:
        return direct
    if os.name == "nt":
        for suffix in _WINDOWS_SHIM_SUFFIXES:
            found = shutil.which(name + suffix)
            if found:
                return found
    return None


def require_tool(name: str, purpose: str) -> str:
    resolved = find_tool(name)
    if resolved is None:
        raise ToolNotFoundError(
            f"{name!r} is not on PATH, and it is needed to {purpose}."
        )
    return resolved


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(
    argv: list[str],
    cwd: Path,
    *,
    echo: bool = True,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run ``argv`` and stream its output straight through.

    Always a list, never ``shell=True``: these argv lists are assembled from JSON
    config, and a shell would turn a config value into shell syntax.

    A missing executable comes back as a normal failing result rather than an
    exception, so a caller running several steps can report "this one is missing" in
    the same shape as "this one failed".
    """
    resolved = find_tool(argv[0])
    real_argv = [resolved or argv[0], *argv[1:]]
    if echo:
        print(f"$ {' '.join(argv)}", flush=True)
    if resolved is None:
        print(f"  not found on PATH: {argv[0]}", file=sys.stderr, flush=True)
        return CommandResult(argv=real_argv, returncode=NOT_FOUND)

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}
    completed = subprocess.run(real_argv, cwd=cwd, check=False, env=merged_env)
    return CommandResult(argv=real_argv, returncode=completed.returncode)


def capture(
    argv: list[str], cwd: Path, *, timeout: float | None = DEFAULT_CAPTURE_TIMEOUT
) -> tuple[int, str]:
    """Run ``argv`` and return ``(returncode, stdout)``, stderr folded in.

    For reading a tool's answer where the two streams are interchangeable — a version
    string, a diagnostic banner. ``errors="replace"`` because a banner in an unexpected
    encoding must not crash a diagnostic command.

    Do **not** use this to parse machine-readable output: see [`capture_streams`].
    """
    code, out, err = capture_streams(argv, cwd, timeout=timeout)
    return code, out + err


def capture_streams(
    argv: list[str], cwd: Path, *, timeout: float | None = DEFAULT_CAPTURE_TIMEOUT
) -> tuple[int, str, str]:
    """Run ``argv`` and return ``(returncode, stdout, stderr)`` kept apart, as text.

    Required whenever stdout is parsed: a warning written to stderr, folded into
    stdout, silently corrupts a machine-readable record.

    Decoding is lossy (``errors="replace"``), which is right for a diagnostic banner and
    wrong for anything whose exact bytes matter. Use [`capture_bytes`] for those.
    """
    code, out, err = capture_bytes(argv, cwd, timeout=timeout)
    return (
        code,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


def capture_bytes(
    argv: list[str], cwd: Path, *, timeout: float | None = DEFAULT_CAPTURE_TIMEOUT
) -> tuple[int, bytes, bytes]:
    """Run ``argv`` and return its raw output, undecoded.

    The caller decides what a byte sequence that is not valid UTF-8 means.

    ``TIMED_OUT`` is returned rather than raised so a caller that runs several probes
    reports a hung tool in the same shape as a missing or failing one.
    """
    resolved = find_tool(argv[0])
    if resolved is None:
        return NOT_FOUND, b"", b""
    try:
        completed = subprocess.run(
            [resolved, *argv[1:]],
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        return TIMED_OUT, expired.stdout or b"", b""
    return completed.returncode, completed.stdout or b"", completed.stderr or b""
