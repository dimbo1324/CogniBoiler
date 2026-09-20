"""The liveness file: what the container healthcheck sees, and what it must not see.

A healthcheck that passes while the service is deaf is worse than none at all, so the
awkward cases here are the ones where the file exists but should not count.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import pytest
from cogniboiler_runtime.liveness import (
    DEFAULT_MAX_AGE_S,
    LivenessFile,
    is_fresh,
    main,
)


class TestBeat:
    def test_a_healthy_service_writes_the_moment_it_beat(self, tmp_path: Path) -> None:
        path = tmp_path / "live"
        before = int(time.time())
        LivenessFile(path).beat(healthy=True)
        assert path.exists()
        assert int(path.read_text(encoding="ascii")) >= before

    def test_an_unhealthy_service_leaves_the_file_where_it_was(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "live"
        liveness = LivenessFile(path)
        liveness.beat(healthy=True)
        stamp = path.read_text(encoding="ascii")
        time.sleep(0.01)
        liveness.beat(healthy=False)
        # Not touched: the file ages out and the healthcheck fails, which is the point.
        assert path.read_text(encoding="ascii") == stamp

    def test_a_service_that_was_never_healthy_writes_no_file_at_all(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "live"
        LivenessFile(path).beat(healthy=False)
        assert not path.exists()
        assert is_fresh(path, DEFAULT_MAX_AGE_S) is False

    async def test_run_creates_the_directory_and_follows_the_health(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "deeper" / "live"
        healthy = False
        liveness = LivenessFile(path, interval_s=0.001)
        task = asyncio.create_task(liveness.run(lambda: healthy))
        try:
            async with asyncio.timeout(5.0):
                while not path.parent.exists():
                    await asyncio.sleep(0.001)
                assert not path.exists()
                healthy = True
                while not path.exists():
                    await asyncio.sleep(0.001)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestFreshness:
    def test_a_file_older_than_the_window_is_not_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "live"
        path.write_text("0", encoding="ascii")
        old = time.time() - 120
        os.utime(path, (old, old))
        assert is_fresh(path, 30.0) is False
        assert is_fresh(path, 300.0) is True

    def test_a_missing_file_is_not_fresh(self, tmp_path: Path) -> None:
        assert is_fresh(tmp_path / "never-written", DEFAULT_MAX_AGE_S) is False

    def test_the_window_is_read_to_the_second_on_both_sides(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "live"
        path.write_text("0", encoding="ascii")
        inside = time.time() - 29.0
        os.utime(path, (inside, inside))
        assert is_fresh(path, 30.0) is True
        outside = time.time() - 31.0
        os.utime(path, (outside, outside))
        assert is_fresh(path, 30.0) is False


class TestCommandLine:
    def test_the_exit_code_is_what_docker_reads(self, tmp_path: Path) -> None:
        path = tmp_path / "live"
        assert main([str(path)]) == 1
        LivenessFile(path).beat(healthy=True)
        assert main([str(path)]) == 0
        # The same file, aged past the window the healthcheck was given.
        stale = time.time() - 60
        os.utime(path, (stale, stale))
        assert main([str(path)]) == 1
        assert main([str(path), "--max-age-s", "120"]) == 0

    def test_a_directory_in_place_of_the_file_is_reported_as_dead(
        self, tmp_path: Path
    ) -> None:
        # The mount exists but nothing ever wrote the file: a stat succeeds on the
        # directory and its mtime is fresh, so only its kind tells them apart.
        directory = tmp_path / "live"
        directory.mkdir()
        assert is_fresh(directory, DEFAULT_MAX_AGE_S) is False
        assert main([str(directory)]) == 1


class TestAFileThatCannotBeWritten:
    def test_the_service_keeps_beating_and_says_why_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A path whose parent is a file, not a directory: every write fails.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="ascii")
        liveness = LivenessFile(blocker / "live")
        with caplog.at_level(logging.WARNING, logger="cogniboiler_runtime.liveness"):
            for _ in range(5):
                liveness.beat(healthy=True)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "cannot be written" in warnings[0].getMessage()

    def test_a_zero_interval_is_refused_instead_of_spinning(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="interval_s"):
            LivenessFile(tmp_path / "live", interval_s=0.0)
