"""Request values the upstreams cannot take are refused at the edge, as 422s.

A value past an int32 or int64 field, or a time Flux cannot hold, used to become a 500
with an ERROR traceback while the request message was built, or a misleading 503.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import FastAPI
from gateway_fakes import FakeHistorianClient
from httpx import AsyncClient
from influxdb_client.rest import ApiException
from prometheus_client import REGISTRY
from urllib3.exceptions import ReadTimeoutError

INT32_MAX = 2**31 - 1
AFTER_2100_MS = 4_102_444_800_001


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


def historian(app: FastAPI) -> FakeHistorianClient:
    client: FakeHistorianClient = app.state.historian_client
    return client


def historian_failures(code: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "gateway_upstream_failures_total", {"service": "Historian", "code": code}
        )
        or 0.0
    )


def errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.ERROR]


class TestIntegerBounds:
    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/api/v1/alarms/history", {"offset": INT32_MAX + 1}),
            ("/api/v1/alarms/history", {"from_ms": 10**20}),
            ("/api/v1/alarms/history", {"to_ms": AFTER_2100_MS}),
            (f"/api/v1/alarms/{INT32_MAX + 1}", {}),
            ("/api/v1/history", {"start_ms": AFTER_2100_MS}),
            ("/api/v1/history", {"end_ms": 10**20}),
            ("/api/v1/kpi", {"start_ms": 10**20}),
            ("/api/v1/kpi", {"end_ms": AFTER_2100_MS}),
            ("/api/v1/simulation/runs", {"offset": INT32_MAX + 1}),
        ],
    )
    async def test_a_value_past_its_bound_is_422_without_an_error_log(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        caplog: pytest.LogCaptureFixture,
        path: str,
        params: dict[str, Any],
    ) -> None:
        response = await client.get(path, params=params, headers=bearer(viewer_tokens))
        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid"
        assert errors(caplog) == []

    async def test_acknowledging_an_alarm_id_past_int32_is_422(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            f"/api/v1/alarms/{INT32_MAX + 1}/ack",
            json={"comment": "seen"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        ("path", "params"),
        [
            ("/api/v1/alarms/history", {"offset": INT32_MAX, "to_ms": 4102444800000}),
            ("/api/v1/simulation/runs", {"offset": INT32_MAX}),
        ],
    )
    async def test_the_bounds_themselves_are_accepted(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        path: str,
        params: dict[str, Any],
    ) -> None:
        response = await client.get(path, params=params, headers=bearer(viewer_tokens))
        assert response.status_code == 200


class TestHistoryFields:
    @pytest.mark.parametrize(
        "fields",
        [
            'pressure_pa") |> drop(',
            "Pressure",
            ",".join(f"f{i}" for i in range(33)),
            "1abc",
        ],
    )
    async def test_bad_field_lists_are_422(
        self, client: AsyncClient, viewer_tokens: dict[str, str], fields: str
    ) -> None:
        response = await client.get(
            "/api/v1/history", params={"fields": fields}, headers=bearer(viewer_tokens)
        )
        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid_fields"

    async def test_valid_fields_reach_the_historian(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/history",
            params={"fields": "pressure_pa, steam_temp_k"},
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 200
        assert ("fields", ("pressure_pa", "steam_temp_k")) in historian(app).calls

    async def test_the_history_range_is_checked_by_the_route_too(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/history",
            params={"start_ms": 2_000, "end_ms": 1_000},
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid_range"


class TestHistorianErrors:
    @pytest.mark.parametrize("path", ["/api/v1/history", "/api/v1/kpi"])
    @pytest.mark.parametrize(
        ("error", "status", "code", "label"),
        [
            (
                ApiException(status=400, reason="Bad Request"),
                422,
                "request.invalid",
                "HTTP_400",
            ),
            (
                ApiException(status=500, reason="Internal"),
                503,
                "upstream.unavailable",
                "HTTP_500",
            ),
            (
                ReadTimeoutError(None, "/api/v2/query", "Read timed out."),  # type: ignore[arg-type]
                504,
                "upstream.timeout",
                "TIMEOUT",
            ),
            (
                ConnectionRefusedError("refused"),
                503,
                "upstream.unavailable",
                "UNAVAILABLE",
            ),
        ],
    )
    async def test_each_failure_has_its_answer_and_is_counted(
        self,
        app: FastAPI,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        path: str,
        error: Exception,
        status: int,
        code: str,
        label: str,
    ) -> None:
        historian(app).error = error
        before = historian_failures(label)
        response = await client.get(path, headers=bearer(viewer_tokens))
        assert response.status_code == status
        body = response.json()
        assert (body["code"], body["service"]) == (code, "Historian")
        assert historian_failures(label) == before + 1
