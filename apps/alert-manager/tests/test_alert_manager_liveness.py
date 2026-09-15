"""The liveness file behind the alert manager's container healthcheck."""

from __future__ import annotations

import os
import time
from pathlib import Path

from alert_manager.liveness import LivenessFile, is_fresh, main
from alert_manager.subscriber import AlertSubscriber


def test_a_healthy_beat_leaves_a_fresh_file(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    LivenessFile(path).beat(healthy=True)
    assert is_fresh(path, max_age_s=30)


def test_an_unhealthy_beat_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    LivenessFile(path).beat(healthy=False)
    assert not path.exists()
    assert not is_fresh(path, max_age_s=30)


def test_a_file_older_than_the_limit_is_not_fresh(tmp_path: Path) -> None:
    path = tmp_path / "alive"
    path.write_text("0", encoding="ascii")
    long_ago = time.time() - 120
    os.utime(path, (long_ago, long_ago))
    assert not is_fresh(path, max_age_s=30)


def test_the_healthcheck_command_fails_until_the_first_healthy_beat(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alive"
    assert main([str(path)]) == 1
    LivenessFile(path).beat(healthy=True)
    assert main([str(path), "--max-age-s", "30"]) == 0


def test_a_new_subscriber_is_not_connected() -> None:
    assert AlertSubscriber().connected is False
