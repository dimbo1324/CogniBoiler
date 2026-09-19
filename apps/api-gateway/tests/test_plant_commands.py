"""PLC commands and status, KPIs, readiness and the platform view, error bodies."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import cogniboiler_pb2 as pb2
import pytest
from api_gateway.dependencies import get_db
from api_gateway.problems import ProblemError, problem_response
from api_gateway.realtime.hub import Channel, RealtimeHub
from fastapi import FastAPI
from gateway_fakes import FakeHistorianClient, FakePLCClient, plc_status
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


def plc(app: FastAPI) -> FakePLCClient:
    client: FakePLCClient = app.state.plc_client
    return client


async def audit_rows(
    client: AsyncClient, admin: dict[str, str], endpoint: str
) -> list[dict[str, Any]]:
    response = await client.get(
        "/api/v1/audit", params={"endpoint": endpoint}, headers=bearer(admin)
    )
    rows: list[dict[str, Any]] = response.json()["items"]
    return rows


class TestCommands:
    async def test_a_load_demand_reaches_the_plc_under_the_user(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/load",
            json={"load_w": 180e6},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 200
        assert plc(app).calls[-1] == ("set_load_demand", (180e6, "operator1"))

    @pytest.mark.parametrize("load_w", [-1.0, 300.1e6])
    async def test_a_load_outside_0_to_300_mw_is_refused(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        load_w: float,
    ) -> None:
        response = await client.post(
            "/api/v1/commands/load",
            json={"load_w": load_w},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 422
        assert plc(app).calls == []

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("auto", pb2.ControlMode.AUTO),
            ("manual", pb2.ControlMode.MANUAL),
            ("estop", pb2.ControlMode.ESTOP),
        ],
    )
    async def test_every_mode_maps_to_the_contract(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        mode: str,
        expected: int,
    ) -> None:
        await client.post(
            "/api/v1/commands/mode",
            json={"mode": mode},
            headers=bearer(operator_tokens),
        )
        assert plc(app).calls[-1] == ("set_control_mode", (expected, "operator1"))

    async def test_an_unknown_mode_is_refused(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/mode",
            json={"mode": "turbo"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 422

    async def test_a_valve_command_names_the_operator_and_keeps_spray_unset(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/api/v1/commands/valve",
            json={"fuel_valve": 0.4, "feedwater_valve": 0.5, "steam_valve": 0.6},
            headers=bearer(operator_tokens),
        )
        command = plc(app).calls[-1][1]
        assert command.operator_id == "operator1"
        assert command.source == pb2.CommandSource.OPERATOR
        assert not command.HasField("spray_valve")

    async def test_a_valve_command_may_move_the_spray_valve(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/api/v1/commands/valve",
            json={
                "fuel_valve": 0.4,
                "feedwater_valve": 0.5,
                "steam_valve": 0.6,
                "spray_valve": 0.2,
            },
            headers=bearer(operator_tokens),
        )
        command = plc(app).calls[-1][1]
        assert command.HasField("spray_valve")
        assert command.spray_valve == pytest.approx(0.2)

    async def test_setpoints_are_sent_with_the_engineer(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/api/v1/commands/setpoint",
            json={"pressure_pa": 150e5, "water_level_m": 5.0, "steam_temp_k": 800.0},
            headers=bearer(engineer_tokens),
        )
        setpoints = plc(app).calls[-1][1]
        assert (setpoints.pressure_pa, setpoints.water_level_m) == (150e5, 5.0)
        assert setpoints.operator_id == "engineer1"

    async def test_a_reset_is_recorded_under_the_signed_in_user(
        self,
        app: FastAPI,
        client: AsyncClient,
        engineer_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        response = await client.post(
            "/api/v1/commands/reset",
            json={"operator_id": "someone.else"},
            headers=bearer(engineer_tokens),
        )
        assert response.status_code == 200
        assert plc(app).calls[-1] == ("reset_emergency_stop", "engineer1")
        (row,) = await audit_rows(client, admin_tokens, "/api/v1/commands/reset")
        assert row["detail"] == "stated operator_id=someone.else"
        assert row["outcome"] == "accepted"

    async def test_a_reset_without_a_body_is_accepted(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/reset", headers=bearer(engineer_tokens)
        )
        assert response.status_code == 200
        assert plc(app).calls[-1] == ("reset_emergency_stop", "engineer1")

    async def test_an_operator_cannot_reset_a_trip(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/commands/reset", headers=bearer(operator_tokens)
        )
        assert response.status_code == 403
        assert plc(app).calls == []

    async def test_a_plc_refusal_is_returned_as_is(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        plc(app).accept = False
        plc(app).reason = "E-Stop active: reset first"
        response = await client.post(
            "/api/v1/commands/mode",
            json={"mode": "auto"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 200
        body = response.json()
        assert (body["accepted"], body["reason"]) == (
            False,
            "E-Stop active: reset first",
        )

    @pytest.mark.parametrize(
        ("path", "body", "tokens"),
        [
            ("/api/v1/commands/load", {"load_w": 1e6}, "operator"),
            ("/api/v1/commands/mode", {"mode": "auto"}, "operator"),
            (
                "/api/v1/commands/valve",
                {"fuel_valve": 0.1, "feedwater_valve": 0.1, "steam_valve": 0.1},
                "operator",
            ),
            (
                "/api/v1/commands/setpoint",
                {"pressure_pa": 140e5, "water_level_m": 4.8, "steam_temp_k": 811.0},
                "engineer",
            ),
            ("/api/v1/commands/reset", None, "engineer"),
        ],
    )
    async def test_an_unreachable_plc_is_503(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        engineer_tokens: dict[str, str],
        path: str,
        body: dict[str, Any] | None,
        tokens: str,
    ) -> None:
        plc(app).down = True
        chosen = operator_tokens if tokens == "operator" else engineer_tokens
        response = await client.post(path, json=body, headers=bearer(chosen))
        assert response.status_code == 503
        assert response.json()["service"] == "PLCService"


class TestPlcStatus:
    async def test_the_status_maps_mode_loops_and_conditions(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/plc/status", headers=bearer(viewer_tokens))
        assert response.status_code == 200
        status = response.json()
        assert status["mode"] == "auto"
        assert status["trip_cause"] is None
        assert status["latest_command"]["source"] == "pid"
        assert status["loops"][0]["name"] == "pressure"
        assert status["active_conditions"][0]["severity"] == "warning"
        assert status["reset_blockers"] == ["not tripped"]
        assert status["active_setpoints"]["pressure_pa"] == 139e5

    async def test_a_latched_trip_names_its_cause(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        plc(app).status = plc_status(
            mode=pb2.ControlMode.ESTOP,
            emergency_stop_active=True,
            active_trip=pb2.SafetyEventMsg(
                parameter="pressure_pa", value=190e5, threshold=185e5, timestamp_ms=9
            ),
        )
        response = await client.get("/api/v1/plc/status", headers=bearer(viewer_tokens))
        status = response.json()
        assert status["mode"] == "estop"
        assert status["trip_cause"] == {
            "parameter": "pressure_pa",
            "value": 190e5,
            "threshold": 185e5,
            "timestamp_ms": 9,
        }

    async def test_an_unreachable_plc_is_503(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        plc(app).down = True
        response = await client.get("/api/v1/plc/status", headers=bearer(viewer_tokens))
        assert response.status_code == 503


class TestKpi:
    async def test_ratios_are_ratios_of_means(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        historian: FakeHistorianClient = app.state.historian_client
        historian.kpi_values = {
            ("electrical_power_w", "mean"): 250e6,
            ("electrical_power_w", "count"): 900.0,
            ("fuel_heat_input_w", "mean"): 625e6,
            ("heat_to_cycle_w", "mean"): 560e6,
            ("co2_kg_s", "mean"): 35.0,
            ("nox_ppmv", "mean"): 45.0,
            ("nox_ppmv", "max"): 60.0,
            ("overall_health_pct", "mean"): 97.0,
            ("overall_health_pct", "min"): 95.5,
        }
        response = await client.get(
            "/api/v1/kpi",
            params={"start_ms": 1_000, "end_ms": 901_000},
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 200
        kpi = response.json()
        assert kpi["samples"] == 900
        assert kpi["source"] == "raw"
        assert kpi["net_efficiency"] == pytest.approx(0.4)
        assert kpi["boiler_efficiency"] == pytest.approx(560 / 625)
        assert kpi["plant_heat_rate_j_per_j"] == pytest.approx(2.5)
        assert kpi["turbine_heat_rate_j_per_j"] == pytest.approx(560 / 250)
        assert kpi["co2_intensity_kg_per_j"] == pytest.approx(35.0 / 250e6)
        assert (kpi["peak_nox_ppmv"], kpi["lowest_health_pct"]) == (60.0, 95.5)
        assert historian.calls[-1] == ("fetch_kpi_inputs", (1_000, 901_000))

    async def test_a_unit_that_does_not_generate_has_no_efficiency(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        historian: FakeHistorianClient = app.state.historian_client
        historian.kpi_values = {
            ("electrical_power_w", "mean"): 0.5e6,
            ("fuel_heat_input_w", "mean"): 40e6,
            ("heat_to_cycle_w", "mean"): 30e6,
        }
        kpi = (await client.get("/api/v1/kpi", headers=bearer(viewer_tokens))).json()
        assert kpi["mean_electrical_power_w"] == 0.5e6
        assert kpi["net_efficiency"] is None
        assert kpi["plant_heat_rate_j_per_j"] is None
        assert kpi["boiler_efficiency"] == pytest.approx(0.75)

    async def test_no_data_is_all_null(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        kpi = (await client.get("/api/v1/kpi", headers=bearer(viewer_tokens))).json()
        assert kpi["samples"] == 0
        assert kpi["net_efficiency"] is None
        assert kpi["mean_nox_ppmv"] is None

    @pytest.mark.parametrize(
        ("params", "code"),
        [
            ({"start_ms": 10, "end_ms": 10}, "request.invalid_range"),
            ({"start_ms": 0, "end_ms": 91 * 86_400_000}, "request.range_too_long"),
        ],
    )
    async def test_ranges_are_validated(
        self,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        params: dict[str, int],
        code: str,
    ) -> None:
        response = await client.get(
            "/api/v1/kpi", params=params, headers=bearer(viewer_tokens)
        )
        assert response.status_code == 422
        assert response.json()["code"] == code

    async def test_an_unreachable_historian_is_503(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        app.state.historian_client.up = False
        response = await client.get("/api/v1/kpi", headers=bearer(viewer_tokens))
        assert response.status_code == 503
        assert response.json()["service"] == "Historian"


class TestReadiness:
    async def test_everything_up_is_ready(self, client: AsyncClient) -> None:
        response = await client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        names = {item["name"]: item for item in body["components"]}
        assert set(names) == {
            "database",
            "physics-engine",
            "plc-controller",
            "alert-manager",
            "historian",
        }
        assert names["database"]["required"] is True
        assert all(item["latency_ms"] is not None for item in body["components"])

    async def test_a_down_upstream_degrades_but_stays_ready(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        plc(app).down = True
        app.state.historian_client.up = False
        response = await client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        down = {item["name"] for item in body["components"] if item["state"] == "down"}
        assert down == {"plc-controller", "historian"}

    async def test_no_database_is_not_ready(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        async def broken_database() -> AsyncGenerator[Any]:
            raise ConnectionError("database is down")
            yield None

        app.dependency_overrides[get_db] = broken_database
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"

    async def test_the_platform_view_adds_the_realtime_hub(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        hub = RealtimeHub(queue_size=4, max_rate_hz=10.0)
        hub.register()
        hub.publish(Channel.TELEMETRY, "state", {"x": 1})
        app.state.realtime_hub = hub
        response = await client.get("/api/v1/platform", headers=bearer(viewer_tokens))
        body = response.json()
        assert body["websocket_clients"] == 1
        assert body["telemetry_age_s"] is not None
        assert body["readiness"]["status"] == "ready"

    async def test_the_platform_view_without_a_hub(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        body = (
            await client.get("/api/v1/platform", headers=bearer(viewer_tokens))
        ).json()
        assert (body["websocket_clients"], body["telemetry_age_s"]) == (0, None)


class TestProblemDetails:
    async def test_an_unknown_route_is_a_problem_document(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/v1/nowhere")
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/problem+json"
        body = response.json()
        assert body["code"] == "http.404"
        assert body["type"] == "urn:cogniboiler:problem:http.404"
        assert body["instance"] == "/api/v1/nowhere"

    async def test_validation_errors_list_locations_not_values(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/auth/login", json={"username": "x", "password": "secret-value-1"}
        )
        body = response.json()
        assert body["code"] == "request.invalid"
        assert all(
            set(error) == {"location", "message", "type"} for error in body["errors"]
        )

    async def test_an_unexpected_error_is_a_500_without_details(
        self, app: FastAPI, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def explode() -> None:
            raise RuntimeError("database password is hunter2")

        app.add_api_route("/boom", explode)
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/boom")
        assert response.status_code == 500
        assert response.json()["code"] == "server.error"
        assert "hunter2" not in response.text
        assert "Unhandled error on GET /boom" in caplog.text

    def test_an_unknown_status_gets_a_generic_title(self) -> None:
        request = Request({"type": "http", "path": "/x", "headers": []})
        response = problem_response(request, 599, "odd", "odd status")
        assert b'"title":"Error"' in response.body

    def test_extra_members_never_replace_the_standard_ones(self) -> None:
        error = ProblemError(409, "x", "detail", extra={"status": 200, "hint": "h"})
        request = Request({"type": "http", "path": "/x", "headers": []})
        response = problem_response(
            request, error.status, error.code, error.detail, extra=error.extra
        )
        assert b'"status":409' in response.body
        assert b'"hint":"h"' in response.body
