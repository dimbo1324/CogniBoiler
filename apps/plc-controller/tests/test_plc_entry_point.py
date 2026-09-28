"""The command line and the wiring of `python -m plc_controller`."""

from __future__ import annotations

from typing import Any

import pytest
from plc_controller import __main__ as entry
from plc_controller.server import listen_address


class TestCommandLine:
    def test_the_defaults_keep_every_port_on_this_machine(self) -> None:
        args = entry.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 50051
        assert args.physics_target == "localhost:50052"
        assert (args.mqtt_host, args.mqtt_port) == ("localhost", 1883)
        assert (args.metrics_host, args.metrics_port) == ("127.0.0.1", 9102)

    def test_main_passes_the_parsed_values_to_serve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        served: dict[str, Any] = {}
        logging_for: list[str] = []

        async def serve(**kwargs: Any) -> None:
            served.update(kwargs)

        monkeypatch.setattr(entry, "serve", serve)
        monkeypatch.setattr(entry, "configure_logging", logging_for.append)
        monkeypatch.setenv("MQTT_USERNAME", "plc")
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        code = entry.main(
            ["--host", "0.0.0.0", "--port", "6000", "--metrics-port", "0"]
        )
        assert code == 0
        assert logging_for == ["plc-controller"]
        assert (served["host"], served["port"]) == ("0.0.0.0", 6000)
        assert served["metrics_port"] == 0
        assert (served["mqtt_username"], served["mqtt_password"]) == ("plc", None)


class TestListenAddress:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1", "127.0.0.1:50051"),
            ("0.0.0.0", "0.0.0.0:50051"),
            ("::", "[::]:50051"),
            ("[::1]", "[::1]:50051"),
            ("localhost", "localhost:50051"),
        ],
    )
    def test_an_ipv6_host_is_bracketed(self, host: str, expected: str) -> None:
        assert listen_address(host, 50051) == expected
