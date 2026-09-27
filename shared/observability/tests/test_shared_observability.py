"""Log format, correlation scope and the correlation id's trip through gRPC."""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Iterator

import grpc
import grpc.aio
import pytest
import structlog
from cogniboiler_observability import (
    ServerObservability,
    accepted_correlation_id,
    client_interceptors,
    configure_logging,
    correlation_scope,
    current_correlation_id,
)
from prometheus_client import REGISTRY

METHOD = "/test.Echo/CorrelationId"


@pytest.fixture
def captured_logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    configure_logging("test-service", level="INFO", fmt="json")
    yield stream
    root.handlers[:] = handlers
    root.setLevel(level)


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_a_log_line_is_one_json_object_with_the_agreed_fields(
    captured_logs: io.StringIO,
) -> None:
    with correlation_scope("abc-123"):
        logging.getLogger("plc_controller.service").info(
            "Mode %s -> %s", "auto", "estop"
        )

    [line] = _lines(captured_logs)
    assert line["event"] == "Mode auto -> estop"
    assert line["level"] == "info"
    assert line["service"] == "test-service"
    assert line["logger"] == "plc_controller.service"
    assert line["correlation_id"] == "abc-123"
    assert str(line["timestamp"]).endswith("Z")


def test_text_stays_readable_utf8(captured_logs: io.StringIO) -> None:
    logging.getLogger("physics").info("speed 10×, 140 bar")
    assert "speed 10×, 140 bar" in captured_logs.getvalue()


def test_outside_a_request_the_correlation_id_is_null(
    captured_logs: io.StringIO,
) -> None:
    logging.getLogger("scan").warning("stream lost")
    [line] = _lines(captured_logs)
    assert line["correlation_id"] is None
    assert line["level"] == "warning"


def test_an_exception_is_rendered_into_the_line(captured_logs: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("x").exception("failed")
    [line] = _lines(captured_logs)
    assert "ValueError: boom" in str(line["exception"])


def test_a_scope_adopts_a_safe_id_and_replaces_an_unsafe_one() -> None:
    with correlation_scope("request-1") as adopted:
        assert adopted == "request-1"
        assert current_correlation_id() == "request-1"
    assert current_correlation_id() is None
    with correlation_scope("has space\nand newline") as replaced:
        assert replaced != "has space\nand newline"
    assert accepted_correlation_id("x" * 129) is None
    assert accepted_correlation_id(None) is None


def _identity(value: bytes) -> bytes:
    return value


async def _echo(request: bytes, context: grpc.aio.ServicerContext) -> bytes:
    return (current_correlation_id() or "").encode()


async def _refuse(request: bytes, context: grpc.aio.ServicerContext) -> bytes:
    await context.abort(grpc.StatusCode.NOT_FOUND, "no such thing")
    return b""


async def test_the_correlation_id_travels_in_grpc_metadata() -> None:
    server = grpc.aio.server(interceptors=[ServerObservability()])
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "test.Echo",
                {
                    "CorrelationId": grpc.unary_unary_rpc_method_handler(
                        _echo,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    "Refuse": grpc.unary_unary_rpc_method_handler(
                        _refuse,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{port}", interceptors=client_interceptors()
        ) as channel:
            echo = channel.unary_unary(
                METHOD, request_serializer=_identity, response_deserializer=_identity
            )
            with correlation_scope("from-the-gateway"):
                assert await echo(b"") == b"from-the-gateway"
            started_here = (await echo(b"")).decode()
            assert len(started_here) == 32

            refuse = channel.unary_unary(
                "/test.Echo/Refuse",
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            with pytest.raises(grpc.aio.AioRpcError):
                await refuse(b"")
    finally:
        await server.stop(grace=None)

    assert (
        REGISTRY.get_sample_value(
            "grpc_server_handled_total",
            {"grpc_method": METHOD, "grpc_code": "OK"},
        )
        == 2.0
    )
    assert (
        REGISTRY.get_sample_value(
            "grpc_server_handled_total",
            {"grpc_method": "/test.Echo/Refuse", "grpc_code": "NOT_FOUND"},
        )
        == 1.0
    )


@pytest.fixture
def console_logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[io.StringIO]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    configure_logging("test-service", level="INFO", fmt="console")
    yield stream
    root.handlers[:] = handlers
    root.setLevel(level)


class TestRedaction:
    def test_a_sensitive_key_is_masked_in_the_json_line(
        self, captured_logs: io.StringIO
    ) -> None:
        structlog.get_logger("api_gateway.auth").info(
            "login attempt",
            username="operator",
            password="hunter2",
            refresh_token="abc.def",
            Authorization="Bearer xyz",
            session_cookie="c=1",
            api_key="k-123",
            client_secret="s3",
        )
        [line] = _lines(captured_logs)
        assert line["username"] == "operator"
        for key in (
            "password",
            "refresh_token",
            "Authorization",
            "session_cookie",
            "api_key",
            "client_secret",
        ):
            assert line[key] == "***", key
        for secret in ("hunter2", "abc.def", "xyz", "c=1", "k-123", "s3"):
            assert secret not in captured_logs.getvalue()

    def test_a_sensitive_key_inside_a_nested_value_is_masked(
        self, captured_logs: io.StringIO
    ) -> None:
        structlog.get_logger("x").info(
            "request", headers={"Authorization": "Bearer xyz", "Accept": "json"}
        )
        [line] = _lines(captured_logs)
        assert line["headers"] == {"Authorization": "***", "Accept": "json"}

    def test_the_standard_fields_are_left_alone(
        self, captured_logs: io.StringIO
    ) -> None:
        with correlation_scope("abc-123"):
            logging.getLogger("x").info("token refreshed")
        [line] = _lines(captured_logs)
        assert line["event"] == "token refreshed"
        assert line["correlation_id"] == "abc-123"

    def test_the_console_is_masked_too(self, console_logs: io.StringIO) -> None:
        structlog.get_logger("x").info("login", password="hunter2")
        assert "hunter2" not in console_logs.getvalue()


class TestConsoleEscaping:
    def test_a_newline_in_a_value_cannot_forge_a_second_line(
        self, console_logs: io.StringIO
    ) -> None:
        logging.getLogger("x").info(
            "login by %s", "bob\n2026-09-27 [info] admin logged in\r"
        )
        lines = console_logs.getvalue().splitlines()
        assert len(lines) == 1
        assert "bob\\n2026-09-27 [info] admin logged in\\r" in lines[0]

    def test_terminal_control_sequences_are_escaped(
        self, console_logs: io.StringIO
    ) -> None:
        structlog.get_logger("x").info("comment", text="\x1b[2Jcleared\x00")
        output = console_logs.getvalue()
        assert "\x1b" not in output
        assert "\x00" not in output
        assert "\\x1b[2Jcleared\\x00" in output

    def test_readable_text_is_left_as_it_is(self, console_logs: io.StringIO) -> None:
        logging.getLogger("physics").info("speed 10×, 140 bar\tnominal")
        assert "speed 10×, 140 bar\tnominal" in console_logs.getvalue()

    def test_a_traceback_keeps_its_lines(self, console_logs: io.StringIO) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("x").exception("failed")
        output = console_logs.getvalue()
        assert "Traceback" in output
        assert "ValueError: boom" in output
        assert len(output.splitlines()) > 2
