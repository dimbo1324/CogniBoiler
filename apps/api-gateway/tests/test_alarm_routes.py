"""Alarm routes: lists, history, details and acknowledgements through AlarmService."""

from __future__ import annotations

from typing import Any

import cogniboiler_pb2 as pb2
import pytest
from api_gateway.routers.alarms import alarm_from_proto
from fastapi import FastAPI
from gateway_fakes import FakeAlarmClient, alarm, not_found
from httpx import AsyncClient


def bearer(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access']}"}


def alarms(app: FastAPI) -> FakeAlarmClient:
    client: FakeAlarmClient = app.state.alarm_client
    return client


async def audit_outcome(client: AsyncClient, admin: dict[str, str], path: str) -> Any:
    response = await client.get(
        "/api/v1/audit", params={"endpoint": path}, headers=bearer(admin)
    )
    return response.json()["items"][0]["outcome"]


class TestMapping:
    @pytest.mark.parametrize(
        ("state", "name", "acknowledged", "cleared"),
        [
            (pb2.AlarmState.ALARM_ACTIVE_UNACK, "ACTIVE_UNACK", False, False),
            (pb2.AlarmState.ALARM_ACTIVE_ACK, "ACTIVE_ACK", True, False),
            (pb2.AlarmState.ALARM_CLEARED_UNACK, "CLEARED_UNACK", False, True),
            (pb2.AlarmState.ALARM_CLEARED, "CLEARED", True, True),
            (pb2.AlarmState.ALARM_STATE_UNSPECIFIED, "CLEARED", True, True),
        ],
    )
    def test_states_and_flags(
        self, state: int, name: str, acknowledged: bool, cleared: bool
    ) -> None:
        mapped = alarm_from_proto(alarm(3, state=state))
        assert mapped.state == name
        assert mapped.acknowledged is acknowledged
        assert mapped.cleared is cleared

    def test_unset_times_and_names_become_null(self) -> None:
        mapped = alarm_from_proto(alarm(3))
        assert mapped.alarm_id == "3"
        assert mapped.cleared_at_ms is None
        assert mapped.acknowledged_at_ms is None
        assert mapped.acknowledged_by is None
        assert mapped.ack_comment is None
        assert mapped.occurred_at_ms == mapped.raised_at_ms

    def test_an_acknowledgement_is_carried(self) -> None:
        mapped = alarm_from_proto(
            alarm(
                3,
                state=pb2.AlarmState.ALARM_ACTIVE_ACK,
                acknowledged_at_ms=5,
                acknowledged_by="operator1",
                ack_comment="seen",
                cleared_at_ms=9,
            )
        )
        assert (mapped.acknowledged_at_ms, mapped.acknowledged_by) == (5, "operator1")
        assert (mapped.ack_comment, mapped.cleared_at_ms) == ("seen", 9)


class TestReading:
    async def test_the_list_asks_for_open_alarms_when_told(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/alarms",
            params={"active_only": "true", "limit": 20},
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 200
        assert [item["id"] for item in response.json()] == [7, 8]
        name, request = alarms(app).calls[-1]
        assert name == "list_alarms"
        assert request.open_only is True
        assert request.limit == 20

    async def test_history_passes_every_filter_and_returns_the_total(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/alarms/history",
            params={
                "severity": "critical",
                "parameter": "water_level_m",
                "from_ms": 100,
                "to_ms": 200,
                "limit": 10,
                "offset": 30,
            },
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 200
        page = response.json()
        assert (page["total"], page["limit"], page["offset"]) == (42, 10, 30)
        request = alarms(app).calls[-1][1]
        assert request.severity == "critical"
        assert request.parameter == "water_level_m"
        assert (request.from_ms, request.to_ms) == (100, 200)
        assert (request.limit, request.offset) == (10, 30)
        assert request.open_only is False

    async def test_history_refuses_an_unknown_severity(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get(
            "/api/v1/alarms/history",
            params={"severity": "info"},
            headers=bearer(viewer_tokens),
        )
        assert response.status_code == 422

    async def test_a_detail_lists_every_transition(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/alarms/7", headers=bearer(viewer_tokens))
        assert response.status_code == 200
        detail = response.json()
        assert detail["alarm"]["id"] == 7
        first, second = detail["transitions"]
        assert first["from_state"] is None
        assert first["to_state"] == "ACTIVE_UNACK"
        assert first["comment"] is None
        assert (second["from_state"], second["to_state"]) == (
            "ACTIVE_UNACK",
            "ACTIVE_ACK",
        )
        assert (second["actor"], second["comment"]) == ("operator1", "seen")

    async def test_an_unknown_alarm_is_404(
        self, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        response = await client.get("/api/v1/alarms/99", headers=bearer(viewer_tokens))
        assert response.status_code == 404
        assert response.json()["code"] == "alarms.not_found"

    @pytest.mark.parametrize(
        "path", ["/api/v1/alarms", "/api/v1/alarms/history", "/api/v1/alarms/7"]
    )
    async def test_an_unreachable_alarm_service_is_503_without_its_message(
        self,
        app: FastAPI,
        client: AsyncClient,
        viewer_tokens: dict[str, str],
        path: str,
    ) -> None:
        alarms(app).down = True
        response = await client.get(path, headers=bearer(viewer_tokens))
        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "upstream.unavailable"
        assert body["service"] == "AlarmService"
        assert "connection refused" not in response.text

    async def test_reading_needs_a_token(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/alarms")
        assert response.status_code == 401


class TestAcknowledging:
    async def test_the_operator_name_and_comment_reach_the_service(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        response = await client.post(
            "/api/v1/alarms/7/ack",
            json={"comment": "drum level checked"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True
        assert response.json()["alarms"][0]["state"] == "ACTIVE_ACK"
        assert alarms(app).calls[-1] == (
            "acknowledge",
            (7, "operator1", "drum level checked"),
        )
        outcome = await audit_outcome(client, admin_tokens, "/api/v1/alarms/7/ack")
        assert outcome == "acknowledged 1 alarm(s)"

    async def test_a_refusal_is_a_200_with_the_reason_and_audited(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        admin_tokens: dict[str, str],
    ) -> None:
        alarms(app).accept = False
        alarms(app).reason = "already acknowledged"
        response = await client.post(
            "/api/v1/alarms/7/ack", json={}, headers=bearer(operator_tokens)
        )
        assert response.status_code == 200
        assert response.json() | {"timestamp_ms": 0} == {
            "accepted": False,
            "reason": "already acknowledged",
            "timestamp_ms": 0,
            "alarms": [],
        }
        outcome = await audit_outcome(client, admin_tokens, "/api/v1/alarms/7/ack")
        assert outcome == "refused: already acknowledged"

    async def test_all_alarms_of_one_severity(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/alarms/ack-all",
            json={"comment": "shift handover", "severity": "warning"},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 200
        assert len(response.json()["alarms"]) == 2
        assert alarms(app).calls[-1] == (
            "acknowledge_all",
            ("operator1", "shift handover", "warning"),
        )

    async def test_all_alarms_without_a_severity(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        await client.post(
            "/api/v1/alarms/ack-all", json={}, headers=bearer(operator_tokens)
        )
        assert alarms(app).calls[-1] == ("acknowledge_all", ("operator1", "", ""))

    async def test_a_viewer_cannot_acknowledge(
        self, app: FastAPI, client: AsyncClient, viewer_tokens: dict[str, str]
    ) -> None:
        for path in ("/api/v1/alarms/7/ack", "/api/v1/alarms/ack-all"):
            response = await client.post(path, json={}, headers=bearer(viewer_tokens))
            assert response.status_code == 403
        assert alarms(app).calls == []

    async def test_a_comment_is_limited_to_500_characters(
        self, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/alarms/7/ack",
            json={"comment": "x" * 501},
            headers=bearer(operator_tokens),
        )
        assert response.status_code == 422

    async def test_acknowledging_an_unknown_alarm_is_404(
        self, app: FastAPI, client: AsyncClient, operator_tokens: dict[str, str]
    ) -> None:
        async def missing(*_: object) -> pb2.AcknowledgeResult:
            raise not_found()

        alarms(app).acknowledge = missing  # type: ignore[method-assign]
        response = await client.post(
            "/api/v1/alarms/99/ack", json={}, headers=bearer(operator_tokens)
        )
        assert response.status_code == 404

    @pytest.mark.parametrize("path", ["/api/v1/alarms/7/ack", "/api/v1/alarms/ack-all"])
    async def test_an_unreachable_service_is_503(
        self,
        app: FastAPI,
        client: AsyncClient,
        operator_tokens: dict[str, str],
        path: str,
    ) -> None:
        alarms(app).down = True
        response = await client.post(path, json={}, headers=bearer(operator_tokens))
        assert response.status_code == 503
