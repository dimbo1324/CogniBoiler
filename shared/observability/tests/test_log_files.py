"""Log files: where they go, their format, rotation, and a directory that cannot be used."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from cogniboiler_observability import configure_logging
from cogniboiler_observability.logs import ServiceLogFile


@pytest.fixture(autouse=True)
def _restore_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("LOG_DIR", "LOG_FILE_MAX_BYTES", "LOG_FILE_BACKUPS", "LOG_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if isinstance(handler, ServiceLogFile):
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def _file_handlers() -> list[ServiceLogFile]:
    return [h for h in logging.getLogger().handlers if isinstance(h, ServiceLogFile)]


def _json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def test_the_file_gets_json_lines_whatever_the_console_shows(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    configure_logging("historian", fmt="console", log_dir=tmp_path / "logs")
    logging.getLogger("historian.writer").warning("flush took %.1f s", 2.5)

    [line] = _json_lines(tmp_path / "logs" / "historian.log")
    assert line["event"] == "flush took 2.5 s"
    assert line["service"] == "historian"
    assert line["logger"] == "historian.writer"
    assert line["level"] == "warning"
    assert str(line["timestamp"]).endswith("Z")
    out = capsys.readouterr().out
    assert "flush took 2.5 s" in out
    assert not out.lstrip().startswith("{")


def test_the_directory_comes_from_the_environment(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "nested" / "logs"))
    configure_logging("plc-controller")
    logging.getLogger("scan").info("Mode auto -> estop")
    [line] = _json_lines(tmp_path / "nested" / "logs" / "plc-controller.log")
    assert line["event"] == "Mode auto -> estop"
    assert json.loads(capsys.readouterr().out)["event"] == "Mode auto -> estop"


@pytest.mark.parametrize("value", ["", "   "])
def test_without_a_directory_there_is_no_file(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LOG_DIR", value)
    configure_logging("physics-engine")
    assert _file_handlers() == []
    logging.getLogger("x").info("still on stdout")
    assert "still on stdout" in capsys.readouterr().out


def test_text_stays_readable_utf8_in_the_file(tmp_path: Path) -> None:
    configure_logging("physics-engine", log_dir=tmp_path)
    logging.getLogger("physics").info("speed 10×, 140 bar")
    assert "speed 10×, 140 bar" in (tmp_path / "physics-engine.log").read_text("utf-8")


def test_an_exception_reaches_the_file(tmp_path: Path) -> None:
    configure_logging("alert-manager", log_dir=tmp_path)
    try:
        raise RuntimeError("database gone")
    except RuntimeError:
        logging.getLogger("alarms").exception("write failed")
    [line] = _json_lines(tmp_path / "alert-manager.log")
    assert "RuntimeError: database gone" in str(line["exception"])


def test_files_rotate_by_size_and_keep_a_bounded_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_FILE_MAX_BYTES", "400")
    monkeypatch.setenv("LOG_FILE_BACKUPS", "2")
    configure_logging("api-gateway", log_dir=tmp_path)
    for number in range(60):
        logging.getLogger("requests").info("request %03d served", number)

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["api-gateway.log", "api-gateway.log.1", "api-gateway.log.2"]
    for path in tmp_path.iterdir():
        assert path.stat().st_size <= 400
    last = _json_lines(tmp_path / "api-gateway.log")[-1]
    assert last["event"] == "request 059 served"


def test_a_directory_that_cannot_be_used_leaves_stdout_working(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    occupied = tmp_path / "logs"
    occupied.write_text("a file, not a directory", encoding="utf-8")
    configure_logging("opcua-server", log_dir=occupied)

    assert _file_handlers() == []
    [warning] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert warning["level"] == "warning"
    assert str(warning["event"]).startswith(
        f"Log files are off: cannot write to {occupied}"
    )
    logging.getLogger("opcua").info("serving")
    assert json.loads(capsys.readouterr().out)["event"] == "serving"


def test_configuring_again_replaces_and_closes_the_previous_file(
    tmp_path: Path,
) -> None:
    configure_logging("historian", log_dir=tmp_path / "first")
    [first] = _file_handlers()
    configure_logging("historian", log_dir=tmp_path / "second")
    [second] = _file_handlers()
    assert second is not first
    assert first.stream is None
    logging.getLogger("x").info("only in the second file")
    assert (tmp_path / "first" / "historian.log").read_text("utf-8") == ""
    assert _json_lines(tmp_path / "second" / "historian.log")[0]["event"] == (
        "only in the second file"
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LOG_FILE_MAX_BYTES", "ten", "LOG_FILE_MAX_BYTES must be a whole number"),
        ("LOG_FILE_BACKUPS", "-1", "LOG_FILE_BACKUPS must not be negative"),
    ],
)
def test_a_malformed_limit_is_refused_at_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        configure_logging("historian", log_dir=tmp_path)
