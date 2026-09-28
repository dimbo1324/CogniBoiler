"""
Errors as Problem Details (RFC 9457).

Every error the gateway returns is `application/problem+json` with the standard members
`type`, `title`, `status`, `detail` and `instance`, plus a stable machine-readable `code`
that clients branch on instead of parsing text. Validation errors add `errors` with the
location and message of each problem — never the submitted value, which may be a
password. Upstream failures never echo gRPC or driver messages to the caller: the log
gets the status code.

A failed gRPC call maps to one answer by its status code: DEADLINE_EXCEEDED is 504
`upstream.timeout` (the upstream may still have acted, so the audit outcome says
"unknown"), INVALID_ARGUMENT is 422 `request.invalid` (retrying cannot help), anything
else is 503 `upstream.unavailable`. Each is counted in gateway_upstream_failures_total.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

import grpc
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from influxdb_client.rest import ApiException
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from api_gateway.audit import set_audit_outcome
from api_gateway.observability import UPSTREAM_FAILURES

HISTORIAN = "Historian"

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:cogniboiler:problem:"


class ProblemDetails(BaseModel):
    """An error body (RFC 9457); members beyond these depend on the code."""

    model_config = ConfigDict(extra="allow")

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str = Field(..., description="Stable, machine-readable error code.")


def _problem(description: str) -> dict[str, Any]:
    # FastAPI documents a model under application/json; name the real media type too.
    schema = {"$ref": "#/components/schemas/ProblemDetails"}
    return {
        "model": ProblemDetails,
        "description": description,
        "content": {PROBLEM_MEDIA_TYPE: {"schema": schema}},
    }


# The answers every route that calls an upstream service can give besides its own.
UPSTREAM_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: _problem(
        "upstream.unavailable: the upstream service (named in `service`) failed."
    ),
    504: _problem(
        "upstream.timeout: the upstream service did not answer in time; a command "
        "may still have been carried out."
    ),
}


class ProblemError(Exception):
    """An error answered with a Problem Details body."""

    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        *,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})


def _upstream_failure(service: str, code: str, status: int) -> ProblemError:
    """Count and log a failed upstream call and build its answer (503, 504 or 422)."""
    UPSTREAM_FAILURES.labels(service, code).inc()
    logger.warning("%s call failed: %s", service, code)
    extra = {"service": service}
    if status == 504:
        return ProblemError(
            504, "upstream.timeout", f"{service} did not answer in time.", extra=extra
        )
    if status == 422:
        return ProblemError(
            422, "request.invalid", f"{service} refused the request.", extra=extra
        )
    return ProblemError(
        503, "upstream.unavailable", f"{service} is unavailable.", extra=extra
    )


def rpc_status_code(exc: grpc.RpcError) -> grpc.StatusCode:
    code = getattr(exc, "code", None)
    value = code() if callable(code) else None
    return value if isinstance(value, grpc.StatusCode) else grpc.StatusCode.UNKNOWN


def upstream_problem(
    request: Request,
    service: str,
    exc: grpc.RpcError,
    *,
    not_found: ProblemError | None = None,
) -> ProblemError:
    """The Problem, metric, log line and audit outcome of one failed gRPC call."""
    code = rpc_status_code(exc)
    if code is grpc.StatusCode.DEADLINE_EXCEEDED:
        set_audit_outcome(request, "unknown: upstream deadline exceeded")
    else:
        set_audit_outcome(request, f"failed: {code.name}")
    if code is grpc.StatusCode.NOT_FOUND and not_found is not None:
        return not_found
    return _upstream_failure(service, code.name, _GRPC_STATUS.get(code, 503))


_GRPC_STATUS: dict[grpc.StatusCode, int] = {
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    grpc.StatusCode.INVALID_ARGUMENT: 422,
}


@asynccontextmanager
async def historian_call() -> AsyncIterator[None]:
    """Run an InfluxDB query; a failure leaves as a Problem, as upstream_call's do.

    InfluxDB answers 400 to a query it cannot run, which retrying cannot fix: 422.
    """
    try:
        yield
    except ApiException as exc:
        status = 422 if exc.status == 400 else 503
        raise _upstream_failure(HISTORIAN, f"HTTP_{exc.status}", status) from exc
    except (Urllib3TimeoutError, TimeoutError) as exc:
        raise _upstream_failure(HISTORIAN, "TIMEOUT", 504) from exc
    except (OSError, Urllib3HTTPError) as exc:
        raise _upstream_failure(HISTORIAN, "UNAVAILABLE", 503) from exc


@asynccontextmanager
async def upstream_call(
    request: Request, service: str, *, not_found: ProblemError | None = None
) -> AsyncIterator[None]:
    """Run gRPC calls; a failure leaves as the Problem `upstream_problem` picks."""
    try:
        yield
    except grpc.RpcError as exc:
        raise upstream_problem(request, service, exc, not_found=not_found) from exc


def _title(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def problem_response(
    request: Request,
    status: int,
    code: str,
    detail: str,
    *,
    headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"{PROBLEM_TYPE_PREFIX}{code}",
        "title": _title(status),
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
    }
    if extra:
        body.update({key: value for key, value in extra.items() if key not in body})
    return JSONResponse(
        body,
        status_code=status,
        headers=dict(headers or {}),
        media_type=PROBLEM_MEDIA_TYPE,
    )


async def _problem_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ProblemError):
        raise exc
    return problem_response(
        request,
        exc.status,
        exc.code,
        exc.detail,
        headers=exc.headers,
        extra=exc.extra,
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    detail = exc.detail if isinstance(exc.detail, str) else _title(exc.status_code)
    return problem_response(
        request,
        exc.status_code,
        f"http.{exc.status_code}",
        detail,
        headers=exc.headers,
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
    errors = [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()
    ]
    return problem_response(
        request,
        422,
        "request.invalid",
        "The request did not pass validation.",
        extra={"errors": errors},
    )


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    return problem_response(
        request, 500, "server.error", "An unexpected error occurred."
    )


def install_problem_handlers(app: FastAPI) -> None:
    """Answer every error of the application with Problem Details."""
    app.add_exception_handler(ProblemError, _problem_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)
