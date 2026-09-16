"""
Errors as Problem Details (RFC 9457).

Every error the gateway returns is `application/problem+json` with the standard members
`type`, `title`, `status`, `detail` and `instance`, plus a stable machine-readable `code`
that clients branch on instead of parsing text. Validation errors add `errors` with the
location and message of each problem — never the submitted value, which may be a
password. Upstream failures never echo gRPC or driver messages to the caller: those go
to the log.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:cogniboiler:problem:"


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


def upstream_unavailable(service: str, exc: BaseException) -> ProblemError:
    """503 for a failed upstream call; the cause is logged, not returned."""
    logger.warning("%s call failed: %s", service, exc)
    return ProblemError(
        503,
        "upstream.unavailable",
        f"{service} is unavailable.",
        extra={"service": service},
    )


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
