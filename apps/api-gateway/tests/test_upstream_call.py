"""One mapping from a failed gRPC call to a Problem, a metric and an audit outcome."""

from __future__ import annotations

import logging
from typing import Any

import grpc
import grpc.aio
import pytest
from api_gateway.problems import ProblemError, upstream_call
from gateway_fakes import UpstreamDownError
from prometheus_client import REGISTRY
from starlette.requests import Request


def rpc_error(code: grpc.StatusCode, details: str = "secret detail") -> grpc.RpcError:
    return grpc.aio.AioRpcError(
        code, grpc.aio.Metadata(), grpc.aio.Metadata(), details=details
    )


def a_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/x", "headers": []})


def outcome(request: Request) -> Any:
    return request.scope.get("state", {}).get("audit_outcome")


def failures(service: str, code: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "gateway_upstream_failures_total", {"service": service, "code": code}
        )
        or 0.0
    )


async def failing_call(
    request: Request, error: grpc.RpcError, **options: Any
) -> ProblemError:
    with pytest.raises(ProblemError) as raised:
        async with upstream_call(request, "PLCService", **options):
            raise error
    return raised.value


@pytest.mark.parametrize(
    ("code", "status", "problem", "audit"),
    [
        (
            grpc.StatusCode.UNAVAILABLE,
            503,
            "upstream.unavailable",
            "failed: UNAVAILABLE",
        ),
        (
            grpc.StatusCode.DEADLINE_EXCEEDED,
            504,
            "upstream.timeout",
            "unknown: upstream deadline exceeded",
        ),
        (
            grpc.StatusCode.INVALID_ARGUMENT,
            422,
            "request.invalid",
            "failed: INVALID_ARGUMENT",
        ),
        (grpc.StatusCode.INTERNAL, 503, "upstream.unavailable", "failed: INTERNAL"),
    ],
)
async def test_each_status_code_has_its_problem_and_audit_outcome(
    code: grpc.StatusCode, status: int, problem: str, audit: str
) -> None:
    request = a_request()
    before = failures("PLCService", code.name)
    error = await failing_call(request, rpc_error(code))
    assert (error.status, error.code) == (status, problem)
    assert error.extra == {"service": "PLCService"}
    assert "secret detail" not in error.detail
    assert outcome(request) == audit
    assert failures("PLCService", code.name) == before + 1


async def test_not_found_becomes_the_given_problem_and_is_no_failure() -> None:
    request = a_request()
    before = failures("PLCService", "NOT_FOUND")
    missing = ProblemError(404, "alarms.not_found", "The alarm does not exist.")
    error = await failing_call(
        request, rpc_error(grpc.StatusCode.NOT_FOUND), not_found=missing
    )
    assert error is missing
    assert failures("PLCService", "NOT_FOUND") == before
    assert outcome(request) == "failed: NOT_FOUND"


async def test_not_found_without_a_mapping_is_an_outage() -> None:
    error = await failing_call(a_request(), rpc_error(grpc.StatusCode.NOT_FOUND))
    assert error.status == 503


async def test_an_error_without_a_code_counts_as_unknown() -> None:
    class Bare(grpc.RpcError):
        pass

    request = a_request()
    before = failures("PLCService", "UNKNOWN")
    error = await failing_call(request, Bare())
    assert error.status == 503
    assert failures("PLCService", "UNKNOWN") == before + 1


async def test_the_log_names_the_code_not_the_error_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="api_gateway.problems"):
        await failing_call(a_request(), UpstreamDownError())
    (record,) = caplog.records
    assert "UNAVAILABLE" in record.getMessage()
    assert "connection refused" not in record.getMessage()


async def test_a_successful_call_passes_through() -> None:
    request = a_request()
    async with upstream_call(request, "PLCService"):
        pass
    assert outcome(request) is None
