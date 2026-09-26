"""Turning the raw captures of the README spec into the files the README shows.

The planning here — which frames make the GIF, how long each stays, which size to try
next — is plain Python and tested without images. Only the two ``encode_*`` functions
touch Pillow, and they import it themselves, so this module loads under any Python: the
scripts' own tests run under the interpreter the orchestrator was started with, which
does not have Pillow.

Every file has to fit the repository's large-file limit. The encoder therefore takes an
ordered list of attempts, each a width and a palette size, and keeps the first result
that fits: a sharper picture when it is small enough, a smaller one when it is not.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BYTES_PER_KIB = 1024


class MediaError(RuntimeError):
    """A manifest, an attempt list or an encoding that cannot produce a README file."""


@dataclass(frozen=True, slots=True)
class Frame:
    file: str
    at_ms: int
    caption: str
    hold_ms: int
    key: bool


@dataclass(frozen=True, slots=True)
class Attempt:
    width: int
    colors: int

    def describe(self) -> str:
        return f"{self.width} px, {self.colors} colours"


def load_frames(text: str) -> list[Frame]:
    """The frame manifest the Playwright spec writes, checked field by field."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise MediaError(f"the frame manifest is not JSON: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise MediaError("the frame manifest holds no frames")
    frames: list[Frame] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MediaError(f"frame {index} is not an object")
        try:
            frame = Frame(
                file=_text(item, "file"),
                at_ms=_whole(item, "at_ms"),
                caption=_text(item, "caption"),
                hold_ms=_whole(item, "hold_ms"),
                key=_flag(item, "key"),
            )
        except MediaError as error:
            raise MediaError(f"frame {index}: {error}") from error
        if "/" in frame.file or "\\" in frame.file or frame.file.startswith("."):
            # A manifest names files next to it, never a path that climbs out of it.
            raise MediaError(f"frame {index}: {frame.file!r} is not a plain file name")
        frames.append(frame)
    return frames


def load_attempts(raw: Any) -> list[Attempt]:
    if not isinstance(raw, list) or not raw:
        raise MediaError("an attempt list must name at least one attempt")
    attempts = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MediaError(f"attempt {index} is not an object")
        width, colors = _whole(item, "width"), _whole(item, "colors")
        if width <= 0:
            raise MediaError(f"attempt {index}: width must be positive")
        if not 2 <= colors <= 256:
            raise MediaError(f"attempt {index}: a palette holds 2 to 256 colours")
        attempts.append(Attempt(width=width, colors=colors))
    return attempts


def select_frames(frames: Sequence[Frame], max_frames: int) -> list[Frame]:
    """At most ``max_frames`` frames, in order, keeping every key frame.

    Key frames are the moments the story is told by — the trip, the reset — so they are
    never dropped, even when they alone exceed the limit. The rest are taken evenly
    across the whole sequence, which keeps a slow stretch from crowding out a fast one.
    """
    if max_frames <= 0:
        raise MediaError("max_frames must be positive")
    keys = [index for index, frame in enumerate(frames) if frame.key]
    others = [index for index, frame in enumerate(frames) if not frame.key]
    room = max(max_frames - len(keys), 0)
    if room >= len(others):
        chosen = set(keys) | set(others)
    elif room == 0:
        chosen = set(keys)
    else:
        step = len(others) / room
        chosen = set(keys) | {others[int(position * step)] for position in range(room)}
    return [frames[index] for index in sorted(chosen)]


def durations(frames: Sequence[Frame], frame_ms: int, last_hold_ms: int) -> list[int]:
    """How long the GIF shows each frame; the last one lingers before the loop."""
    if not frames:
        return []
    shown = [frame.hold_ms or frame_ms for frame in frames]
    shown[-1] = max(shown[-1], last_hold_ms)
    return shown


def fitted_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """The size scaled down to ``target_width``, never up, keeping the aspect ratio."""
    if width <= target_width:
        return width, height
    return target_width, max(1, round(height * target_width / width))


def first_that_fits(
    attempts: Sequence[Attempt],
    encode: Callable[[Attempt], bytes],
    budget_bytes: int,
) -> tuple[Attempt, bytes]:
    """The first attempt whose result fits the budget, or an error naming every size."""
    tried = []
    for attempt in attempts:
        data = encode(attempt)
        if len(data) <= budget_bytes:
            return attempt, data
        tried.append(f"{attempt.describe()}: {len(data) / BYTES_PER_KIB:.0f} KiB")
    raise MediaError(
        f"nothing fits {budget_bytes / BYTES_PER_KIB:.0f} KiB — " + "; ".join(tried)
    )


def encode_png(source: Path, attempt: Attempt) -> bytes:
    """A screenshot as an indexed-colour PNG: exact for flat UI, a fraction of the size."""
    from PIL import Image

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    image = image.resize(
        fitted_size(image.width, image.height, attempt.width), Image.Resampling.LANCZOS
    )
    indexed = image.quantize(colors=attempt.colors, dither=Image.Dither.NONE)
    buffer = io.BytesIO()
    indexed.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_gif(
    sources: Sequence[Path], shown_ms: Sequence[int], attempt: Attempt
) -> bytes:
    """Frames as one looping GIF with a palette shared by every frame.

    One palette for the whole animation keeps a colour from flickering between frames,
    and lets Pillow store each frame as the rectangle that changed since the last one —
    on a dashboard that is mostly the mimic's values, so most of every frame is free.
    """
    from PIL import Image

    if len(sources) != len(shown_ms) or not sources:
        raise MediaError(
            "a GIF needs as many durations as frames, and one frame at least"
        )
    frames = []
    for source in sources:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        frames.append(
            image.resize(
                fitted_size(image.width, image.height, attempt.width),
                Image.Resampling.LANCZOS,
            )
        )
    # The palette is taken from a strip of evenly spaced frames, so a colour that only
    # appears later — the red of the trip — has a place in it.
    sample = frames[:: max(1, len(frames) // 6)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for position, image in enumerate(sample):
        strip.paste(image, (0, position * image.height))
    palette = strip.quantize(colors=attempt.colors, dither=Image.Dither.NONE)
    indexed = [
        image.quantize(palette=palette, dither=Image.Dither.NONE) for image in frames
    ]
    buffer = io.BytesIO()
    indexed[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=indexed[1:],
        duration=list(shown_ms),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return buffer.getvalue()


def _text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise MediaError(f"{key!r} must be a non-empty string")
    return value


def _whole(item: dict[str, Any], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MediaError(f"{key!r} must be a whole number of zero or more")
    return value


def _flag(item: dict[str, Any], key: str) -> bool:
    value = item.get(key)
    if not isinstance(value, bool):
        raise MediaError(f"{key!r} must be true or false")
    return value
