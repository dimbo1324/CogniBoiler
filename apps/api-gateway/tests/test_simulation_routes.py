"""Simulation control and the plant snapshot: roles, upstream calls, the run log."""

from __future__ import annotations

import logging
from typing import Any

import cogniboiler_pb2 as pb2
import pytest
from api_gateway import plant_state
from api_gateway.dependencies import get_db
from fastapi import FastAPI
from gateway_fakes import FakePhysicsClient, simulation_status
from httpx import AsyncClient
from sqlalchemy import text


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


def physics(app: FastAPI) -> FakePhysicsClient:
    client: FakePhysicsClient = app.state.physics_client
    return client


async def runs(client: AsyncClient, tokens: dict[str, str]) -> list[dict[str, Any]]:
    response = await client.get("/api/v1/simulation/runs", headers=bearer(tokens))
    assert response.status_code == 200
    items: list[dict[str, Any]] = response.json()["items"]
    return items


class TestReading:
    async def test_the_plant_snapshot_names_faults_and_qualities(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/plant", headers=bearer(viewer_tokens))
        assert response.status_code == 200
        plant = response.json()
        assert plant["boiler"]["quality"] == "good"
        assert plant["timestamp_ms"] == plant["boiler"]["timestamp_ms"]
        assert plant["faults"][0]["kind"] == "sensor_drift"
        assert plant["faults"][0]["label"] == "sensor_drift:drum_level"
        assert plant["sensors"] == [
            {"sensor_id": "drum_level", "quality": "uncertain", "measured_value": 4.9}
        ]
        assert plant["simulation"]["run_state"] == "running"

    async def test_the_simulation_status_maps_a_pause(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        physics(app).status = simulation_status(
            run_state=pb2.SimulationRunState.SIMULATION_PAUSED, speed_factor=10.0
        )
        response = await client.get("/api/v1/simulation", headers=bearer(viewer_tokens))
        status = response.json()
        assert status["run_state"] == "paused"
        assert status["speed_factor"] == 10.0
        assert status["scenario"] == "nominal"

    async def test_the_scenarios_and_the_current_one(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/simulation/scenarios", headers=bearer(viewer_tokens)
        )
        body = response.json()
        assert body["current"] == "nominal"
        assert [item["name"] for item in body["scenarios"]] == ["nominal", "hot_start"]

    @pytest.mark.parametrize(
        "path",
        ["/api/v1/plant", "/api/v1/simulation", "/api/v1/simulation/scenarios"],
    )
    async def test_an_unreachable_physics_engine_is_503(
        self,
        app: FastAPI,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        path: str,
    ) -> None:
        physics(app).down = True
        response = await client.get(path, headers=bearer(viewer_tokens))
        assert response.status_code == 503
        assert response.json()["service"] == "PhysicsService"


class TestControl:
    @pytest.mark.parametrize(
        ("path", "body", "call"),
        [
            ("/api/v1/simulation/pause", None, ("pause", "engineer1")),
            ("/api/v1/simulation/resume", None, ("resume", "engineer1")),
            (
                "/api/v1/simulation/speed",
                {"speed_factor": 10.0},
                ("set_speed", (10.0, "engineer1")),
            ),
            (
                "/api/v1/simulation/step",
                {"steps": 60},
                ("step", (60, "engineer1")),
            ),
        ],
    )
    async def test_an_engineer_controls_the_run_under_their_name(
        self,
        app: FastAPI,
        client: AsyncClient,
        engineer_tokens: dict[str, str],
        path: str,
        body: dict[str, Any] | None,
        call: tuple[str, Any],
    ) -> None:
        response = await client.post(path, json=body, headers=bearer(engineer_tokens))
        assert response.status_code == 200
        assert response.json()["accepted"] is True
        assert response.json()["status"]["run_id"] == 3
        assert physics(app).calls[-1] == call

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/simulation/pause",
            "/api/v1/simulation/resume",
            "/api/v1/simulation/scenario",
            "/api/v1/simulation/faults",
        ],
    )
    async def test_an_operator_cannot_change_the_simulation(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        path: str,
    ) -> None:
        response = await client.post(
            path,
            json={"name": "nominal", "kind": "steam_leak"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 403
        assert physics(app).calls == []

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/api/v1/simulation/speed", {"speed_factor": 0}),
            ("/api/v1/simulation/speed", {"speed_factor": 50.5}),
            ("/api/v1/simulation/step", {"steps": 0}),
            ("/api/v1/simulation/step", {"steps": 3601}),
            ("/api/v1/simulation/scenario", {"name": "Nominal Start"}),
            ("/api/v1/simulation/faults", {"kind": "meteor"}),
            ("/api/v1/simulation/faults", {"kind": "steam_leak", "severity": 1.5}),
            ("/api/v1/simulation/faults", {"kind": "steam_leak", "ramp_s": -1}),
        ],
    )
    async def test_requests_outside_the_limits_never_reach_the_engine(
        self,
        app: FastAPI,
        client: AsyncClient,
        engineer_tokens: dict[str, str],
        path: str,
        body: dict[str, Any],
    ) -> None:
        response = await client.post(path, json=body, headers=bearer(engineer_tokens))
        assert response.status_code == 422
        assert physics(app).calls == []

    async def test_a_refusal_is_a_200_with_its_reason(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        physics(app).accept = False
        physics(app).reason = "stepping needs a paused simulation"
        response = await client.post(
            "/api/v1/simulation/step",
            json={"steps": 5},
            headers=bearer(engineer_tokens),
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is False
        assert response.json()["reason"] == "stepping needs a paused simulation"

    async def test_an_unreachable_engine_is_503_for_a_command(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        physics(app).down = True
        response = await client.post(
            "/api/v1/simulation/pause", headers=bearer(engineer_tokens)
        )
        assert response.status_code == 503


class TestScenarioAndFaultLog:
    async def test_a_loaded_scenario_is_logged_with_its_user(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        physics(app).status = simulation_status(scenario="hot_start", run_id=4)
        response = await client.post(
            "/api/v1/simulation/scenario",
            json={"name": "hot_start"},
            headers=bearer(engineer_tokens),
        )
        assert response.status_code == 200
        assert physics(app).calls[-1] == ("load_scenario", ("hot_start", "engineer1"))
        (row,) = await runs(client, engineer_tokens)
        assert row["kind"] == "scenario"
        assert (row["scenario"], row["run_id"]) == ("hot_start", 4)
        assert row["username"] == "engineer1"
        assert row["fault_label"] is None

    async def test_a_refused_scenario_is_not_logged(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        physics(app).accept = False
        await client.post(
            "/api/v1/simulation/scenario",
            json={"name": "unknown_one"},
            headers=bearer(engineer_tokens),
        )
        assert await runs(client, engineer_tokens) == []

    async def test_an_injected_fault_reaches_the_engine_and_the_log(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/simulation/faults",
            json={
                "kind": "valve_stuck",
                "target": "feedwater",
                "severity": 0.4,
                "ramp_s": 30,
            },
            headers=bearer(engineer_tokens),
        )
        assert response.status_code == 200
        fault = response.json()["faults"][0]
        assert (fault["kind"], fault["target"]) == ("valve_stuck", "feedwater")
        request = next(
            value for name, value in physics(app).calls if name == "inject_fault"
        )
        assert request.kind == pb2.FaultKind.FAULT_VALVE_STUCK
        assert (request.target, request.operator_id) == ("feedwater", "engineer1")
        assert request.severity == pytest.approx(0.4)
        assert request.ramp_s == 30
        (row,) = await runs(client, engineer_tokens)
        assert row["kind"] == "fault_injected"
        assert row["fault_label"] == "fault:feedwater"
        assert row["severity"] == pytest.approx(0.4)

    async def test_a_refused_fault_is_not_logged(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        physics(app).accept = False
        physics(app).reason = "unknown sensor"
        response = await client.post(
            "/api/v1/simulation/faults",
            json={"kind": "sensor_failure", "target": "nowhere"},
            headers=bearer(engineer_tokens),
        )
        assert response.json()["accepted"] is False
        assert await runs(client, engineer_tokens) == []

    async def test_clearing_one_fault_and_every_fault(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        one = await client.delete(
            "/api/v1/simulation/faults/f-1", headers=bearer(engineer_tokens)
        )
        every = await client.delete(
            "/api/v1/simulation/faults", headers=bearer(engineer_tokens)
        )
        assert one.status_code == every.status_code == 200
        requests = [
            value for name, value in physics(app).calls if name == "clear_fault"
        ]
        assert (requests[0].fault_id, requests[0].all) == ("f-1", False)
        assert requests[1].all is True
        assert {request.operator_id for request in requests} == {"engineer1"}
        kinds = [row["kind"] for row in await runs(client, engineer_tokens)]
        assert kinds == ["fault_cleared", "fault_cleared"]

    async def test_the_run_log_pages_newest_first(
        self, app: FastAPI, client: AsyncClient, engineer_tokens: dict[str, str]
    ) -> None:
        for name in ("nominal", "hot_start", "nominal"):
            await client.post(
                "/api/v1/simulation/scenario",
                json={"name": name},
                headers=bearer(engineer_tokens),
            )
        response = await client.get(
            "/api/v1/simulation/runs",
            params={"limit": 2, "offset": 1},
            headers=bearer(engineer_tokens),
        )
        page = response.json()
        assert (page["total"], len(page["items"])) == (3, 2)
        ids = [item["id"] for item in page["items"]]
        assert ids == sorted(ids, reverse=True)

    async def test_an_unrecorded_run_is_logged_and_the_action_still_answers(
        self,
        app: FastAPI,
        client: AsyncClient,
        engineer_tokens: dict[str, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sessions = app.dependency_overrides[get_db]()
        db = await anext(sessions)
        await db.execute(text("DROP TABLE scenario_runs"))
        await db.commit()
        await sessions.aclose()
        with caplog.at_level(logging.ERROR, logger="api_gateway.routers.simulation"):
            response = await client.post(
                "/api/v1/simulation/scenario",
                json={"name": "nominal"},
                headers=bearer(engineer_tokens),
            )
        assert response.status_code == 200
        assert response.json()["accepted"] is True
        assert "scenario_runs NOT stored" in caplog.text


class TestMapping:
    def test_every_fault_kind_has_a_name_both_ways(self) -> None:
        for name, kind in plant_state.FAULT_KINDS_BY_NAME.items():
            mapped = plant_state.fault(pb2.FaultMsg(kind=kind))
            assert mapped.kind == name

    def test_an_unspecified_fault_kind_is_named_so(self) -> None:
        assert plant_state.fault(pb2.FaultMsg()).kind == "unspecified"

    def test_quality_names_are_lower_case(self) -> None:
        assert plant_state.quality_name(pb2.SensorQuality.BAD) == "bad"
