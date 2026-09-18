"""
gRPC interceptors: the correlation id crosses service boundaries in call metadata, and
every server counts and times the calls it handles.

Clients pass `client_interceptors()` to `grpc.aio.insecure_channel`; servers pass
`ServerObservability()` to `grpc.aio.server`. A server adopts the caller's id, or starts
a new one, for exactly the duration of the call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import grpc
import grpc.aio
from prometheus_client import Counter, Histogram

from cogniboiler_observability.correlation import (
    CORRELATION_METADATA_KEY,
    correlation_scope,
    current_correlation_id,
)

GRPC_HANDLED = Counter(
    "grpc_server_handled_total",
    "gRPC calls completed by this server.",
    ["grpc_method", "grpc_code"],
)
GRPC_SECONDS = Histogram(
    "grpc_server_handling_seconds",
    "Time to handle a unary gRPC call.",
    ["grpc_method"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)


def _with_correlation(
    details: grpc.aio.ClientCallDetails,
) -> grpc.aio.ClientCallDetails:
    correlation_id = current_correlation_id()
    if correlation_id is None:
        return details
    metadata = grpc.aio.Metadata()
    for key, value in details.metadata or ():
        metadata.add(key, value)
    metadata.add(CORRELATION_METADATA_KEY, correlation_id)
    return grpc.aio.ClientCallDetails(
        details.method,
        details.timeout,
        metadata,
        details.credentials,
        details.wait_for_ready,
    )


class _UnaryUnaryCorrelation(grpc.aio.UnaryUnaryClientInterceptor):  # type: ignore[misc]
    async def intercept_unary_unary(
        self, continuation: Any, client_call_details: Any, request: Any
    ) -> Any:
        return await continuation(_with_correlation(client_call_details), request)


class _UnaryStreamCorrelation(grpc.aio.UnaryStreamClientInterceptor):  # type: ignore[misc]
    async def intercept_unary_stream(
        self, continuation: Any, client_call_details: Any, request: Any
    ) -> Any:
        return await continuation(_with_correlation(client_call_details), request)


def client_interceptors() -> list[grpc.aio.ClientInterceptor]:
    """Interceptors that put the current correlation id into outgoing metadata."""
    return [_UnaryUnaryCorrelation(), _UnaryStreamCorrelation()]


def _code_name(context: Any, default: str) -> str:
    """The status the handler set, or `default` when it set none."""
    code = context.code() if hasattr(context, "code") else None
    return code.name if isinstance(code, grpc.StatusCode) else default


def _failure_code(context: Any, error: BaseException) -> str:
    if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        return "CANCELLED"
    return _code_name(context, "UNKNOWN")


def _caller_id(details: grpc.HandlerCallDetails) -> str | None:
    for key, value in details.invocation_metadata or ():
        if key == CORRELATION_METADATA_KEY and isinstance(value, str):
            return value
    return None


class ServerObservability(grpc.aio.ServerInterceptor):  # type: ignore[misc]
    """Adopt the caller's correlation id and count and time every call."""

    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[Any]],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        handler = await continuation(handler_call_details)
        if handler is None:
            return None
        method = str(handler_call_details.method)
        caller_id = _caller_id(handler_call_details)

        if handler.unary_unary is not None:
            behavior = handler.unary_unary

            async def unary(request: Any, context: Any) -> Any:
                started = time.perf_counter()
                with correlation_scope(caller_id):
                    try:
                        response = await behavior(request, context)
                    except BaseException as error:
                        GRPC_HANDLED.labels(method, _failure_code(context, error)).inc()
                        raise
                    finally:
                        GRPC_SECONDS.labels(method).observe(
                            time.perf_counter() - started
                        )
                    GRPC_HANDLED.labels(method, _code_name(context, "OK")).inc()
                    return response

            return grpc.unary_unary_rpc_method_handler(
                unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.unary_stream is not None:
            stream_behavior = handler.unary_stream

            async def stream(request: Any, context: Any) -> AsyncIterator[Any]:
                with correlation_scope(caller_id):
                    try:
                        async for item in stream_behavior(request, context):
                            yield item
                    except BaseException as error:
                        GRPC_HANDLED.labels(method, _failure_code(context, error)).inc()
                        raise
                    GRPC_HANDLED.labels(method, _code_name(context, "OK")).inc()

            return grpc.unary_stream_rpc_method_handler(
                stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler
