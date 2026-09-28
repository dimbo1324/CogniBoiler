"""Engineering units, projections, and the API gateway client of the OPC UA server."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import urllib.request

import cogniboiler_pb2 as pb
import pytest
from asyncua import ua
from cogniboiler_observability import correlation_scope
from opcua_fakes import PASSWORD, gateway_server
from opcua_server import gateway
from opcua_server.gateway import (
    GatewayClient,
    GatewaySession,
    GatewayTokens,
    GatewayUnavailableError,
)
from opcua_server.projection import (
    alarm_updates,
    plant_updates,
    plc_updates,
    sensor_qualities,
)
from opcua_server.units import UNECE_NAMESPACE, engineering_units, unit_id


class TestUnits:
    def test_a_unit_id_is_the_code_as_a_big_endian_number(self) -> None:
        assert unit_id("PAL") == (ord("P") << 16) | (ord("A") << 8) | ord("L")
        assert unit_id("59") == (ord("5") << 8) | ord("9")

    def test_a_coded_unit(self) -> None:
        info = engineering_units("Pa")
        assert info is not None
        assert info.NamespaceUri == UNECE_NAMESPACE
        assert info.UnitId == unit_id("PAL")
        assert info.DisplayName.Text == "Pa"
        assert info.Description.Text == "pascal"

    @pytest.mark.parametrize(
        ("unit", "description"),
        [
            ("J/J", "joule per joule"),
            ("kg/MWh", "kilogram per megawatt hour"),
            ("rpm", "rpm"),
        ],
    )
    def test_a_unit_without_a_code_keeps_its_name(
        self, unit: str, description: str
    ) -> None:
        info = engineering_units(unit)
        assert info is not None
        assert info.UnitId == -1
        assert info.Description.Text == description

    def test_text_and_flags_have_no_units(self) -> None:
        assert engineering_units("-") is None


class TestProjections:
    def test_the_plant_status_feeds_valves_kpis_and_simulation(self) -> None:
        message = pb.PlantStatusMsg(
            actuators=pb.ActuatorStateMsg(fuel_valve_command=0.6),
            performance=pb.PerformanceMsg(net_efficiency=0.4),
            health=pb.EquipmentHealthMsg(maintenance_alarm=True),
            simulation=pb.SimulationStatusMsg(
                scenario="nominal",
                run_id=3,
                speed_factor=10.0,
                run_state=pb.SimulationRunState.SIMULATION_PAUSED,
            ),
            active_faults=[pb.FaultMsg(label="z"), pb.FaultMsg(label="a")],
            sensors=[
                pb.SensorStatusMsg(
                    sensor_id="drum_level", quality=pb.SensorQuality.BAD
                ),
                pb.SensorStatusMsg(sensor_id="unknown", quality=pb.SensorQuality.GOOD),
            ],
        )
        updates = dict(plant_updates(message))
        assert updates[2300] == pytest.approx(0.6)
        assert updates[2502] == pytest.approx(0.4)
        assert updates[2557] is True
        assert (updates[2600], updates[2601], updates[2603]) == ("nominal", 3, 10.0)
        assert updates[2604] is True
        assert updates[2605] == ["a", "z"]
        assert updates[2606] == 1

    def test_instrument_qualities_reach_only_known_nodes(self) -> None:
        message = pb.PlantStatusMsg(
            sensors=[
                pb.SensorStatusMsg(
                    sensor_id="drum_level", quality=pb.SensorQuality.BAD
                ),
                pb.SensorStatusMsg(
                    sensor_id="stack_camera", quality=pb.SensorQuality.BAD
                ),
            ]
        )
        assert sensor_qualities(message) == {2101: pb.SensorQuality.BAD}

    def test_the_plc_folder(self) -> None:
        status = pb.PLCStatusMsg(
            mode=pb.ControlMode.ESTOP,
            emergency_stop_active=True,
            active_trip=pb.SafetyEventMsg(
                parameter="pressure_pa", value=190e5, threshold=185e5
            ),
            reset_blockers=["pressure high"],
            setpoints=pb.SetpointsMsg(pressure_pa=140e5),
            warning_count=4,
        )
        updates = dict(plc_updates(status))
        assert updates[2700] == "estop"
        assert updates[2702] == "pressure_pa=1.9e+07 (limit 1.85e+07)"
        assert updates[2704] == ["pressure high"]
        assert (updates[2707], updates[2710], updates[2712]) == (140e5, 4, True)

    def test_an_unknown_control_mode_is_named_unknown(self) -> None:
        updates = dict(plc_updates(pb.PLCStatusMsg(mode=99)))
        assert updates[2700] == "unknown"

    def test_the_open_alarm_count_is_the_total_not_the_page(self) -> None:
        listed = [pb.AlarmMsg(alarm_id=n, severity="warning") for n in (1, 2, 3)]
        updates = dict(alarm_updates(pb.AlarmListMsg(total=250, alarms=listed)))
        assert (updates[2800], len(updates[2803])) == (250, 3)
        unset = dict(alarm_updates(pb.AlarmListMsg(alarms=listed)))
        assert unset[2800] == 3

    def test_no_trip_cause_while_running(self) -> None:
        updates = dict(plc_updates(pb.PLCStatusMsg(mode=pb.ControlMode.AUTO)))
        assert (updates[2700], updates[2702]) == ("auto", "")

    def test_the_alarm_folder_counts_and_lists_open_alarms(self) -> None:
        alarms = pb.AlarmListMsg(
            alarms=[
                pb.AlarmMsg(
                    alarm_id=1,
                    severity="critical",
                    state=pb.AlarmState.ALARM_ACTIVE_UNACK,
                    message="level low-low",
                ),
                pb.AlarmMsg(
                    alarm_id=2,
                    severity="critical",
                    state=pb.AlarmState.ALARM_CLEARED_UNACK,
                    message="pressure high",
                ),
                pb.AlarmMsg(
                    alarm_id=3,
                    severity="warning",
                    state=pb.AlarmState.ALARM_ACTIVE_ACK,
                    message="NOx high",
                ),
                pb.AlarmMsg(alarm_id=4, severity="warning", message="odd"),
            ]
        )
        updates = dict(alarm_updates(alarms))
        assert (updates[2800], updates[2801], updates[2802]) == (4, 2, 1)
        assert updates[2803][0] == "1 | critical | ACTIVE_UNACK | level low-low"
        assert updates[2803][3] == "4 | warning | UNKNOWN | odd"


class TestGatewayClient:
    async def test_sign_in_refresh_and_sign_out(self) -> None:
        with gateway_server() as (url, script):
            client = GatewayClient(url + "/")
            tokens = await client.login("operator1", PASSWORD)
            assert tokens is not None
            assert (tokens.username, tokens.role) == ("operator1", "operator")
            renewed = await client.refresh(tokens.refresh_token)
            assert renewed is not None and renewed.access_token == "access-2"
            await client.logout(renewed.refresh_token)
        assert [call.path for call in script.calls] == [
            "/auth/login",
            "/auth/refresh",
            "/auth/logout",
        ]
        assert script.calls[2].body == {"refresh_token": "refresh-2"}

    async def test_a_refused_sign_in_is_none_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with gateway_server() as (url, _):
            with caplog.at_level(logging.WARNING, logger="opcua_server.gateway"):
                assert await GatewayClient(url).login("operator1", "wrong") is None
        assert "HTTP 401 auth.invalid_credentials" in caplog.text
        assert "wrong" not in caplog.text

    async def test_a_command_carries_the_token_and_the_correlation_id(self) -> None:
        with gateway_server() as (url, script):
            with correlation_scope("call-7"):
                reply = await GatewayClient(url).request(
                    "POST", "/api/v1/commands/load", {"load_w": 1.0}, "access-9"
                )
        assert (reply.status, reply.body) == (200, {"accepted": True, "reason": ""})
        call = script.calls[0]
        assert call.authorization == "Bearer access-9"
        assert call.correlation_id == "call-7"
        assert call.body == {"load_w": 1.0}

    async def test_a_sign_in_names_the_opc_ua_client_to_the_gateway(self) -> None:
        with gateway_server() as (url, script):
            client = GatewayClient(url)
            await client.login("operator1", PASSWORD, client_address="192.0.2.10")
            await client.login("operator1", PASSWORD)
        assert [call.forwarded_for for call in script.calls] == ["192.0.2.10", ""]

    async def test_at_most_a_few_sign_ins_run_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()
        guard = threading.Lock()
        running = [0, 0]

        def blocking(*_: object) -> gateway.GatewayReply:
            with guard:
                running[0] += 1
                running[1] = max(running[1], running[0])
            release.wait(5.0)
            with guard:
                running[0] -= 1
            return gateway.GatewayReply(401, {})

        client = GatewayClient("http://127.0.0.1:1")
        monkeypatch.setattr(client, "_request", blocking)
        logins = [
            asyncio.create_task(client.login("operator1", "pw"))
            for _ in range(gateway.MAX_CONCURRENT_LOGINS + 3)
        ]
        try:
            async with asyncio.timeout(5.0):
                while running[0] < gateway.MAX_CONCURRENT_LOGINS:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert running[1] == gateway.MAX_CONCURRENT_LOGINS
        finally:
            release.set()
            await asyncio.gather(*logins)
            client.close()
        assert running[0] == 0

    async def test_an_error_body_is_returned_with_its_status(self) -> None:
        with gateway_server() as (url, script):
            script.command_status = 422
            script.command_body = {"detail": "load_w out of range"}
            reply = await GatewayClient(url).request("POST", "/api/v1/commands/load")
        assert (reply.status, reply.detail) == (422, "load_w out of range")

    async def test_a_body_that_is_not_a_json_object_is_empty(self) -> None:
        with gateway_server() as (url, script):
            script.raw_command_body = b"<html>proxy error</html>"
            reply = await GatewayClient(url).request("POST", "/api/v1/commands/mode")
        assert reply.body == {}
        assert reply.detail == ""

    async def test_a_reply_cut_short_is_an_unavailable_gateway(self) -> None:
        with gateway_server() as (url, script):
            script.truncate_command_body = True
            with pytest.raises(GatewayUnavailableError, match="IncompleteRead"):
                await GatewayClient(url).request("POST", "/api/v1/commands/load")

    async def test_a_redirect_is_not_followed_with_the_token(self) -> None:
        with gateway_server() as (url, script):
            script.redirect_commands_to = "/elsewhere"
            reply = await GatewayClient(url).request(
                "POST", "/api/v1/commands/load", {"load_w": 1.0}, "access-9"
            )
        assert reply.status == 303
        assert [(call.method, call.path) for call in script.calls] == [
            ("POST", "/api/v1/commands/load")
        ]

    async def test_proxy_variables_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:1")
        with gateway_server() as (url, script):
            client = GatewayClient(url)
            reply = await client.request("POST", "/api/v1/commands/load")
        assert reply.status == 200
        assert len(script.calls) == 1
        # urllib's default ProxyHandler is the one that reads the environment.
        assert not any(
            isinstance(handler, urllib.request.ProxyHandler)
            for handler in client._opener.handlers
        )

    async def test_an_oversized_reply_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        padding = "x" * gateway.MAX_REPLY_BYTES
        with gateway_server() as (url, script):
            script.raw_command_body = json.dumps({"reason": padding}).encode()
            with caplog.at_level(logging.WARNING, logger="opcua_server.gateway"):
                reply = await GatewayClient(url).request(
                    "POST", "/api/v1/commands/load"
                )
        assert (reply.status, reply.body) == (200, {})
        assert "larger than" in caplog.text

    def test_a_non_finite_number_is_never_sent(self) -> None:
        with pytest.raises(ValueError):
            GatewayClient("http://127.0.0.1:1")._request(
                "POST", "/api/v1/commands/load", {"load_w": math.nan}, None
            )

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "ftp://gateway", "api-gateway:8000"]
    )
    def test_only_an_http_gateway_url_is_accepted(self, url: str) -> None:
        with pytest.raises(ValueError, match="http"):
            GatewayClient(url)

    async def test_an_unreachable_gateway_raises(self) -> None:
        with pytest.raises(GatewayUnavailableError, match="POST /auth/login"):
            await GatewayClient("http://127.0.0.1:1").login("operator1", PASSWORD)

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"access_token": "a"},
            {
                **dict.fromkeys(
                    ["access_token", "refresh_token", "username", "role"], "x"
                ),
                "access_expires_at_ms": "soon",
            },
        ],
    )
    def test_incomplete_token_answers_are_rejected(
        self, body: dict[str, object]
    ) -> None:
        assert gateway._tokens(body) is None


def tokens(expires_in_ms: int) -> GatewayTokens:
    return GatewayTokens(
        access_token="a1",
        refresh_token="r1",
        access_expires_at_ms=int(time.time() * 1000) + expires_in_ms,
        username="operator1",
        role="operator",
    )


class FakeClient:
    def __init__(self, refreshed: GatewayTokens | None) -> None:
        self.refreshed = refreshed
        self.refreshes: list[str] = []
        self.logouts: list[str] = []
        self.addresses: list[str | None] = []
        self.logout_fails = False

    async def refresh(
        self, refresh_token: str, client_address: str | None = None
    ) -> GatewayTokens | None:
        self.refreshes.append(refresh_token)
        self.addresses.append(client_address)
        return self.refreshed

    async def logout(
        self, refresh_token: str, client_address: str | None = None
    ) -> None:
        if self.logout_fails:
            raise GatewayUnavailableError("gateway down")
        self.logouts.append(refresh_token)
        self.addresses.append(client_address)


def signed_in(
    result: GatewayTokens | None | Exception,
) -> asyncio.Task[GatewayTokens | None]:
    async def login() -> GatewayTokens | None:
        if isinstance(result, Exception):
            raise result
        return result

    return asyncio.get_running_loop().create_task(login())


class TestGatewaySession:
    async def test_the_sign_in_result_is_used_until_it_nears_expiry(self) -> None:
        client = FakeClient(tokens(900_000))
        session = GatewaySession(client, signed_in(tokens(900_000)))  # type: ignore[arg-type]
        first = await session.tokens()
        assert first is not None and first.access_token == "a1"
        assert client.refreshes == []

    async def test_an_expiring_token_is_refreshed(self) -> None:
        fresh = tokens(900_000)
        client = FakeClient(fresh)
        session = GatewaySession(client, signed_in(tokens(10_000)))  # type: ignore[arg-type]
        assert await session.tokens() is fresh
        assert client.refreshes == ["r1"]

    async def test_a_failed_refresh_closes_the_session(self) -> None:
        client = FakeClient(None)
        session = GatewaySession(client, signed_in(tokens(10_000)))  # type: ignore[arg-type]
        assert await session.tokens() is None
        assert await session.tokens() is None
        assert client.refreshes == ["r1"]

    async def test_a_refused_sign_in_closes_the_session(self) -> None:
        session = GatewaySession(FakeClient(None), signed_in(None))  # type: ignore[arg-type]
        assert await session.tokens() is None

    async def test_an_unreachable_gateway_at_sign_in_is_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = GatewaySession(
            FakeClient(None),  # type: ignore[arg-type]
            signed_in(GatewayUnavailableError("refused")),
        )
        with caplog.at_level(logging.WARNING, logger="opcua_server.gateway"):
            assert await session.tokens() is None
        assert "OPC UA sign-in failed" in caplog.text

    @pytest.mark.parametrize(
        "failure", [GatewayUnavailableError("refused"), RuntimeError("defect")]
    )
    async def test_a_failed_sign_in_closes_the_session_once(
        self, failure: Exception, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = GatewaySession(FakeClient(None), signed_in(failure))  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger="opcua_server.gateway"):
            assert await session.tokens() is None
            assert await session.tokens() is None
        assert caplog.text.count("OPC UA sign-in failed") == 1

    async def test_a_rejected_access_token_forces_a_refresh(self) -> None:
        client = FakeClient(tokens(900_000))
        session = GatewaySession(client, signed_in(tokens(900_000)))  # type: ignore[arg-type]
        await session.tokens()
        await session.invalidate_access()
        await session.tokens()
        assert client.refreshes == ["r1"]

    async def test_invalidating_before_sign_in_changes_nothing(self) -> None:
        session = GatewaySession(FakeClient(None), signed_in(tokens(900_000)))  # type: ignore[arg-type]
        await session.invalidate_access()
        assert await session.tokens() is not None

    async def test_closing_signs_out(self) -> None:
        client = FakeClient(None)
        session = GatewaySession(client, signed_in(tokens(900_000)))  # type: ignore[arg-type]
        await session.tokens()
        await session.close()
        assert client.logouts == ["r1"]
        assert await session.tokens() is None

    async def test_refresh_and_sign_out_name_the_opc_ua_client(self) -> None:
        client = FakeClient(tokens(900_000))
        session = GatewaySession(
            client,  # type: ignore[arg-type]
            signed_in(tokens(10_000)),
            client_address="192.0.2.10",
        )
        await session.tokens()
        await session.close()
        assert session.client_address == "192.0.2.10"
        assert client.addresses == ["192.0.2.10", "192.0.2.10"]

    async def test_closing_twice_signs_out_once(self) -> None:
        client = FakeClient(None)
        session = GatewaySession(client, signed_in(tokens(900_000)))  # type: ignore[arg-type]
        await session.tokens()
        await session.close()
        await session.close()
        assert client.logouts == ["r1"]

    async def test_closing_twice_after_a_refresh_never_signs_out_the_stale_token(
        self,
    ) -> None:
        renewed = GatewayTokens(
            access_token="a2",
            refresh_token="r2",
            access_expires_at_ms=int(time.time() * 1000) + 900_000,
            username="operator1",
            role="operator",
        )
        client = FakeClient(renewed)
        session = GatewaySession(client, signed_in(tokens(10_000)))  # type: ignore[arg-type]
        assert await session.tokens() is renewed
        await session.close()
        await session.close()
        assert client.logouts == ["r2"]

    async def test_closing_after_an_unused_sign_in_still_signs_out(self) -> None:
        client = FakeClient(None)
        login = signed_in(tokens(900_000))
        await asyncio.sleep(0)
        session = GatewaySession(client, login)  # type: ignore[arg-type]
        await login
        await session.close()
        assert client.logouts == ["r1"]

    async def test_closing_during_the_sign_in_signs_out_once_it_finishes(
        self,
    ) -> None:
        answered = asyncio.Event()

        async def slow() -> GatewayTokens | None:
            await answered.wait()
            return tokens(900_000)

        client = FakeClient(None)
        login = asyncio.get_running_loop().create_task(slow())
        session = GatewaySession(client, login)  # type: ignore[arg-type]
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert client.logouts == []
        answered.set()
        await closing
        assert not login.cancelled()
        assert client.logouts == ["r1"]
        assert await session.tokens() is None

    async def test_closing_after_a_failed_sign_in_asks_nothing(self) -> None:
        client = FakeClient(None)
        session = GatewaySession(
            client,  # type: ignore[arg-type]
            signed_in(GatewayUnavailableError("refused")),
        )
        await session.close()
        assert client.logouts == []

    async def test_a_sign_out_without_the_gateway_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeClient(None)
        client.logout_fails = True
        session = GatewaySession(client, signed_in(tokens(900_000)))  # type: ignore[arg-type]
        await session.tokens()
        with caplog.at_level(logging.INFO, logger="opcua_server.gateway"):
            await session.close()
        assert "Gateway sign-out skipped" in caplog.text


def test_status_values_are_typed() -> None:
    from opcua_server.ua_types import node_id, status

    assert node_id(2100, 2) == ua.NodeId(2100, 2)
    assert status(0).is_good()
