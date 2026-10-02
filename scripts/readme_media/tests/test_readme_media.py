"""What readme-media decides without looking at an image: frames, durations and sizes.

Run with:  python -m unittest discover -s scripts -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts._toolkit.config import load_config
from scripts._toolkit.processes import CommandResult
from scripts.readme_media import __main__ as readme_media
from scripts.readme_media.__main__ import SCRIPT_DIR, capture_commands
from scripts.readme_media.media import (
    Attempt,
    Frame,
    MediaError,
    durations,
    first_that_fits,
    fitted_size,
    load_attempts,
    load_frames,
    select_frames,
)


def frame(index: int, *, key: bool = False, hold_ms: int = 0) -> Frame:
    return Frame(
        file=f"frame-{index:04d}.png",
        at_ms=index * 1500,
        caption=f"frame {index}",
        hold_ms=hold_ms,
        key=key,
    )


def manifest(*frames: dict[str, object]) -> str:
    return json.dumps(list(frames))


GOOD = {
    "file": "frame-0000.png",
    "at_ms": 0,
    "caption": "300 MW",
    "hold_ms": 0,
    "key": True,
}


class TestManifest(unittest.TestCase):
    def test_a_manifest_the_spec_writes_is_read_back(self) -> None:
        (read,) = load_frames(manifest(GOOD))
        self.assertEqual(read, Frame("frame-0000.png", 0, "300 MW", 0, True))

    def test_nothing_to_animate_is_refused(self) -> None:
        with self.assertRaisesRegex(MediaError, "no frames"):
            load_frames("[]")
        with self.assertRaisesRegex(MediaError, "not JSON"):
            load_frames("{broken")

    def test_a_field_of_the_wrong_kind_is_named(self) -> None:
        for field, value in [
            ("hold_ms", -1),
            ("hold_ms", 2.5),
            ("at_ms", True),
            ("key", "yes"),
            ("caption", ""),
        ]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(MediaError, field):
                    load_frames(manifest({**GOOD, field: value}))

    def test_a_frame_cannot_name_a_file_outside_the_capture(self) -> None:
        for name in ["../secret.png", "sub/frame.png", "..\\frame.png", ".hidden.png"]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(MediaError, "plain file name"):
                    load_frames(manifest({**GOOD, "file": name}))


class TestSelection(unittest.TestCase):
    def test_a_short_sequence_is_kept_whole(self) -> None:
        frames = [frame(index) for index in range(5)]
        self.assertEqual(select_frames(frames, 40), frames)

    def test_a_long_sequence_is_thinned_evenly_and_keeps_its_order(self) -> None:
        frames = [frame(index) for index in range(100)]
        chosen = select_frames(frames, 10)
        self.assertEqual(len(chosen), 10)
        indexes = [item.at_ms // 1500 for item in chosen]
        self.assertEqual(indexes, sorted(indexes))
        self.assertEqual(indexes[0], 0)
        gaps = {
            later - earlier
            for earlier, later in zip(indexes, indexes[1:], strict=False)
        }
        self.assertEqual(gaps, {10})

    def test_every_key_frame_survives_the_thinning(self) -> None:
        frames = [frame(index, key=index in (17, 63, 99)) for index in range(100)]
        chosen = select_frames(frames, 8)
        self.assertEqual(len(chosen), 8)
        self.assertTrue({17, 63, 99} <= {item.at_ms // 1500 for item in chosen})

    def test_more_key_frames_than_room_keeps_the_story_rather_than_the_limit(
        self,
    ) -> None:
        frames = [frame(index, key=True) for index in range(6)]
        self.assertEqual(select_frames(frames, 3), frames)

    def test_a_limit_of_nothing_is_a_mistake(self) -> None:
        with self.assertRaises(MediaError):
            select_frames([frame(0)], 0)


class TestDurations(unittest.TestCase):
    def test_a_held_moment_keeps_its_hold_and_the_rest_share_the_pace(self) -> None:
        shown = durations([frame(0), frame(1, hold_ms=2000), frame(2)], 320, 0)
        self.assertEqual(shown, [320, 2000, 320])

    def test_the_last_frame_lingers_before_the_loop(self) -> None:
        self.assertEqual(durations([frame(0), frame(1)], 320, 3000), [320, 3000])
        # A hold longer than the linger is not shortened.
        self.assertEqual(durations([frame(0, hold_ms=5000)], 320, 3000), [5000])
        self.assertEqual(durations([], 320, 3000), [])


class TestSizes(unittest.TestCase):
    def test_a_picture_is_scaled_down_but_never_up(self) -> None:
        self.assertEqual(fitted_size(1440, 900, 960), (960, 600))
        self.assertEqual(fitted_size(800, 500, 960), (800, 500))
        self.assertEqual(fitted_size(4000, 1, 100), (100, 1))

    def test_the_first_attempt_that_fits_wins(self) -> None:
        attempts = [Attempt(960, 128), Attempt(800, 64), Attempt(640, 32)]
        sizes = {960: 600, 800: 400, 640: 100}
        tried: list[int] = []

        def encode(attempt: Attempt) -> bytes:
            tried.append(attempt.width)
            return b"x" * sizes[attempt.width]

        chosen, data = first_that_fits(attempts, encode, 450)
        self.assertEqual(chosen, Attempt(800, 64))
        self.assertEqual(len(data), 400)
        # The smaller attempt is never paid for once one fits.
        self.assertEqual(tried, [960, 800])

    def test_nothing_that_fits_names_every_size_it_tried(self) -> None:
        attempts = [Attempt(960, 128), Attempt(640, 32)]
        with self.assertRaisesRegex(MediaError, "960 px.*640 px"):
            first_that_fits(attempts, lambda attempt: b"x" * 2048, 1024)

    def test_attempts_are_checked_before_anything_is_encoded(self) -> None:
        for raw, message in [
            ([], "at least one"),
            ([{"width": 0, "colors": 64}], "width"),
            ([{"width": 800, "colors": 1}], "2 to 256"),
            ([{"width": 800, "colors": 300}], "2 to 256"),
            (["800x64"], "not an object"),
        ]:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(MediaError, message):
                    load_attempts(raw)


class TestConfiguration(unittest.TestCase):
    """The shipped configuration, read the way the script reads it."""

    def setUp(self) -> None:
        self.config = load_config(SCRIPT_DIR, "readme_media.json")

    def test_every_attempt_list_is_valid_and_goes_from_sharp_to_small(self) -> None:
        for raw in (self.config["screenshot_attempts"], self.config["gif"]["attempts"]):
            attempts = load_attempts(raw)
            widths = [attempt.width for attempt in attempts]
            self.assertEqual(widths, sorted(widths, reverse=True))

    def test_the_budget_stays_under_the_repository_limit(self) -> None:
        # pre-commit's check-added-large-files refuses anything over 500 KiB.
        self.assertLess(int(self.config["budget_kib"]), 500)

    def test_every_output_lands_in_docs_images_with_a_distinct_name(self) -> None:
        targets = [shot["target"] for shot in self.config["screenshots"]]
        targets.append(self.config["gif"]["target"])
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(Path(self.config["out_dir"]).as_posix(), "docs/images")

    def test_the_capture_uses_its_own_playwright_config_and_can_skip_the_install(
        self,
    ) -> None:
        with_install = capture_commands(self.config, install=True)
        without = capture_commands(self.config, install=False)
        self.assertEqual(len(with_install), 2)
        self.assertEqual(len(without), 1)
        self.assertIn("--config", without[0])
        self.assertIn(self.config["playwright_config"], without[0])
        self.assertNotIn("playwright.config.ts", without[0])


class TestCaptureFolder(unittest.TestCase):
    """The capture folder is emptied before every run, and it comes from config."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name).resolve() / "repo"
        (self.root / "apps" / "web" / "node_modules").mkdir(parents=True)
        self.config: dict[str, Any] = {
            **load_config(SCRIPT_DIR, "readme_media.json"),
            "web_dir": "apps/web",
        }

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _capture(self) -> bool:
        succeed = mock.Mock(side_effect=lambda argv, *a, **k: CommandResult(argv, 0))
        with (
            mock.patch.object(readme_media, "run", succeed),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return readme_media.capture(
                self.root, self.config, "http://localhost:8080", False
            )

    def test_an_old_frame_is_cleared_before_the_capture(self) -> None:
        raw = self.root / self.config["raw_dir"]
        raw.mkdir(parents=True)
        (raw / "frame-old.png").write_bytes(b"old")
        self.assertTrue(self._capture())
        self.assertTrue(raw.is_dir())
        self.assertEqual(list(raw.iterdir()), [])

    def test_a_folder_outside_the_repository_is_never_deleted(self) -> None:
        outside = self.root.parent / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        self.config["raw_dir"] = "../precious"
        self.assertFalse(self._capture())
        self.assertTrue((outside / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
