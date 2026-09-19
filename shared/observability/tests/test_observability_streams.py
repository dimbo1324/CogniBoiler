"""Streaming calls through the server interceptor, and the metrics endpoint."""

from __future__ import annotations

import asyncio
import logging
import socket
import urllib.request
from collections.abc import AsyncIterator
from typing import Any

import grpc
import grpc.aio
import pytest
from cogniboiler_observability import (
    CORRELATION_METADATA_KEY,
    MQTT_PUBLISHED,
    ServerObservability,
    client_interceptors,
    correlation_scope,
    current_correlation_id,
    start_metrics_server,
)
from prometheus_client import REGISTRY

SERVICE = "test.Streams"


def _identity(value: bytes) -> bytes:
    return value


async def _count(
    request: bytes, context: grpc.aio.ServicerContext
) -> AsyncIterator[bytes]:
    for number in range(int(request or b"3")):
        yield f"{number}:{current_correlation_id()}".encode()


async def _break(
    request: bytes, context: grpc.aio.ServicerContext
) -> AsyncIterator[bytes]:
    yield b"first"
    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "not now")


async def _forever(
    request: bytes, context: grpc.aio.ServicerContext
) -> AsyncIterator[bytes]:
    while True:
        yield b"tick"
        await asyncio.sleep(0.01)


async def _metadata(request: bytes, context: grpc.aio.ServicerContext) -> bytes:
    keys = sorted(key for key, _ in context.invocation_metadata() or ())
    return ",".join(keys).encode()


def handled(method: str, code: str) -> float:
    value = REGISTRY.get_sample_value(
        "grpc_server_handled_total", {"grpc_method": method, "grpc_code": code}
    )
    return value or 0.0


async def served() -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server(interceptors=[ServerObservability()])
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE,
                {
                    "Count": grpc.unary_stream_rpc_method_handler(
                        _count,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    "Break": grpc.unary_stream_rpc_method_handler(
                        _break,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    "Forever": grpc.unary_stream_rpc_method_handler(
                        _forever,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    "Metadata": grpc.unary_unary_rpc_method_handler(
                        _metadata,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, port


def stream(channel: grpc.aio.Channel, name: str) -> Any:
    return channel.unary_stream(
        f"/{SERVICE}/{name}",
        request_serializer=_identity,
        response_deserializer=_identity,
    )


async def test_a_stream_runs_under_the_callers_id_and_is_counted() -> None:
    method = f"/{SERVICE}/Count"
    before = handled(method, "OK")
    server, port = await served()
    try:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{port}", interceptors=client_interceptors()
        ) as channel:
            with correlation_scope("stream-7"):
                items = [item async for item in stream(channel, "Count")(b"3")]
    finally:
        await server.stop(grace=None)
    assert items == [b"0:stream-7", b"1:stream-7", b"2:stream-7"]
    assert handled(method, "OK") == before + 1


async def test_a_stream_that_fails_is_counted_with_its_code() -> None:
    method = f"/{SERVICE}/Break"
    before = handled(method, "FAILED_PRECONDITION")
    server, port = await served()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = stream(channel, "Break")(b"")
            assert await call.read() == b"first"
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call.read()
    finally:
        await server.stop(grace=None)
    assert failed.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert handled(method, "FAILED_PRECONDITION") == before + 1


async def test_a_stream_the_client_leaves_is_counted_as_cancelled() -> None:
    method = f"/{SERVICE}/Forever"
    before = handled(method, "CANCELLED")
    server, port = await served()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = stream(channel, "Forever")(b"")
            assert await call.read() == b"tick"
            call.cancel()
            async with asyncio.timeout(5.0):
                while handled(method, "CANCELLED") == before:
                    await asyncio.sleep(0.01)
    finally:
        await server.stop(grace=None)
    assert handled(method, "CANCELLED") == before + 1


async def test_without_an_id_the_metadata_is_left_alone() -> None:
    server, port = await served()
    try:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{port}", interceptors=client_interceptors()
        ) as channel:
            call = channel.unary_unary(
                f"/{SERVICE}/Metadata",
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            without = await call(b"", metadata=(("x-trace", "1"),))
            with correlation_scope("abc"):
                with_id = await call(b"", metadata=(("x-trace", "1"),))
    finally:
        await server.stop(grace=None)
    assert CORRELATION_METADATA_KEY.encode() not in without
    assert CORRELATION_METADATA_KEY.encode() in with_id
    assert b"x-trace" in with_id


async def test_an_unknown_method_passes_through_the_interceptor() -> None:
    server, port = await served()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(
                f"/{SERVICE}/Nowhere",
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await call(b"")
    finally:
        await server.stop(grace=None)
    assert failed.value.code() == grpc.StatusCode.UNIMPLEMENTED


def test_a_metrics_port_of_zero_disables_the_endpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="cogniboiler_observability.metrics"):
        start_metrics_server(0)
    assert "Metrics endpoint disabled" in caplog.text


def test_the_metrics_endpoint_serves_the_registry() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    MQTT_PUBLISHED.labels("test/topic").inc()
    start_metrics_server(port)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as reply:
        body = reply.read().decode()
    assert 'mqtt_messages_published_total{topic="test/topic"}' in body
