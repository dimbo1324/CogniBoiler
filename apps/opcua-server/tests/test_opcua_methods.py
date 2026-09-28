"""OPC UA methods through the gateway, the PLC and alarm folders, and the MQTT bridge."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import cogniboiler_pb2 as pb
import cogniboiler_pb2_grpc as pb2_grpc
import grpc
import grpc.aio
import pytest
import pytest_asyncio
from aiomqtt import MqttError
from asyncua import ua
from asyncua.crypto.permission_rules import UserRole
from opcua_fakes import PASSWORD, GatewayScript, RecordingOPC, gateway_server
from opcua_server import __main__ as entry
from opcua_server import subscriber, upstreams
from opcua_server.client import AlarmReadClient, PLCStatusClient
from opcua_server.gateway import GatewayClient, GatewaySession, GatewayTokens
from opcua_server.identity import CURRENT_USER, GatewayUser, GatewayUserManager
from opcua_server.methods import MethodHandlers
from opcua_server.subscriber import MQTTOPCBridge
from opcua_server.ua_types import StatusCodes
from prometheus_client import REGISTRY


@pytest.fixture
def gateway() -> Iterator[tuple[str, GatewayScript]]:
    with gateway_server() as found:
        yield found


@pytest_asyncio.fixture
async def as_operator(
    gateway: tuple[str, GatewayScript],
) -> AsyncIterator[GatewayClient]:
    url, _ = gateway
    client = GatewayClient(url)
    user = GatewayUserManager(client).get_user(None, "operator1", PASSWORD)
    token = CURRENT_USER.set(user)
    try:
        yield client
    finally:
        CURRENT_USER.reset(token)
        assert isinstance(user, GatewayUser) and user.session is not None
        await user.session.close()


def outputs(result: Any) -> tuple[bool, str]:
    assert isinstance(result, list), result
    return result[0].Value, result[1].Value


def code(result: Any) -> int:
    assert isinstance(result, ua.StatusCode), result
    return int(result.value)


class TestIdentity:
    def test_an_anonymous_session_is_a_plain_user(self) -> None:
        user = GatewayUserManager(GatewayClient("http://x")).get_user(None)
        assert user is not None
        assert (user.name, user.role) == (None, UserRole.User)

    @pytest.mark.parametrize(("username", "password"), [("", "pw"), ("operator1", "")])
    def test_a_blank_username_or_password_is_refused(
        self, username: str, password: str
    ) -> None:
        manager = GatewayUserManager(GatewayClient("http://x"))
        assert manager.get_user(None, username, password) is None


class TestMethods:
    async def test_a_load_demand_is_sent_as_the_signed_in_user(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript]
    ) -> None:
        _, script = gateway
        result = await MethodHandlers(as_operator).set_load_demand(None, 180e6)
        assert outputs(result) == (True, "")
        login, command = script.calls
        assert login.path == "/auth/login"
        assert (command.path, command.body) == (
            "/api/v1/commands/load",
            {"load_w": 180e6},
        )
        assert command.authorization == "Bearer access-1"
        assert command.correlation_id

    @pytest.mark.parametrize(
        ("call", "path", "body"),
        [
            (
                lambda h: h.set_control_mode(None, ua.Variant(" MANUAL ")),
                "/api/v1/commands/mode",
                {"mode": "manual"},
            ),
            (lambda h: h.reset_emergency_stop(None), "/api/v1/commands/reset", {}),
            (
                lambda h: h.apply_valve_command(None, 0.1, 0.2, ua.Variant(0.3)),
                "/api/v1/commands/valve",
                {"fuel_valve": 0.1, "feedwater_valve": 0.2, "steam_valve": 0.3},
            ),
            (
                lambda h: h.acknowledge_alarm(None, 7, "seen"),
                "/api/v1/alarms/7/ack",
                {"comment": "seen"},
            ),
            (
                lambda h: h.acknowledge_all_alarms(None, "shift"),
                "/api/v1/alarms/ack-all",
                {"comment": "shift"},
            ),
        ],
    )
    async def test_every_method_reaches_its_route(
        self,
        as_operator: GatewayClient,
        gateway: tuple[str, GatewayScript],
        call: Any,
        path: str,
        body: dict[str, Any],
    ) -> None:
        _, script = gateway
        result = await call(MethodHandlers(as_operator))
        assert outputs(result) == (True, "")
        assert (script.calls[-1].path, script.calls[-1].body) == (path, body)

    @pytest.mark.parametrize(
        "call",
        [
            lambda h: h.set_load_demand(None, "a lot"),
            lambda h: h.set_load_demand(None, True),
            lambda h: h.set_control_mode(None, 3),
            lambda h: h.apply_valve_command(None, 0.1, None, 0.3),
            lambda h: h.acknowledge_alarm(None, 0, "x"),
            lambda h: h.acknowledge_alarm(None, 2.5, "x"),
            lambda h: h.acknowledge_alarm(None, 7, None),
            lambda h: h.acknowledge_all_alarms(None, 12),
            lambda h: h.acknowledge_alarm(None, math.nan, "x"),
            lambda h: h.acknowledge_alarm(None, math.inf, "x"),
            lambda h: h.acknowledge_alarm(None, True, "x"),
            lambda h: h.set_load_demand(None, math.nan),
            lambda h: h.set_load_demand(None, -math.inf),
            lambda h: h.apply_valve_command(None, math.inf, 0, 0),
        ],
    )
    async def test_bad_arguments_never_reach_the_gateway(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript], call: Any
    ) -> None:
        _, script = gateway
        result = await call(MethodHandlers(as_operator))
        assert code(result) == StatusCodes.BadInvalidArgument
        # The background sign-in may have run; no command may have.
        assert [c.path for c in script.calls if not c.path.startswith("/auth/")] == []

    @pytest.mark.parametrize("alarm_id", [math.nan, math.inf, -math.inf])
    async def test_an_anonymous_non_finite_alarm_id_is_an_invalid_argument(
        self, alarm_id: float, caplog: pytest.LogCaptureFixture
    ) -> None:
        handlers = MethodHandlers(GatewayClient("http://127.0.0.1:1"))
        with caplog.at_level(logging.DEBUG):
            result = await handlers.acknowledge_alarm(None, ua.Variant(alarm_id), "x")
        assert code(result) == StatusCodes.BadInvalidArgument
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.parametrize(
        ("alarm_id", "path"),
        [
            (7.0, "/api/v1/alarms/7/ack"),
            (2**53 + 1, f"/api/v1/alarms/{2**53 + 1}/ack"),
        ],
    )
    async def test_an_integral_alarm_id_keeps_every_digit(
        self,
        as_operator: GatewayClient,
        gateway: tuple[str, GatewayScript],
        alarm_id: float,
        path: str,
    ) -> None:
        _, script = gateway
        result = await MethodHandlers(as_operator).acknowledge_alarm(None, alarm_id, "")
        assert outputs(result) == (True, "")
        assert script.calls[-1].path == path

    async def test_an_unexpected_failure_is_an_internal_error_not_a_traceback(
        self,
        as_operator: GatewayClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        async def broken(*_: Any, **__: Any) -> Any:
            raise RuntimeError("defect in the forwarding")

        monkeypatch.setattr(as_operator, "request", broken)
        with caplog.at_level(logging.WARNING, logger="opcua_server.methods"):
            result = await MethodHandlers(as_operator).set_load_demand(None, 1e6)
        assert code(result) == StatusCodes.BadInternalError
        assert "SetLoadDemand failed unexpectedly" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    async def test_a_refusal_by_the_plc_is_returned_as_outputs(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript]
    ) -> None:
        _, script = gateway
        script.command_body = {"accepted": False, "reason": "E-Stop active"}
        result = await MethodHandlers(as_operator).set_control_mode(None, "auto")
        assert outputs(result) == (False, "E-Stop active")

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (403, StatusCodes.BadUserAccessDenied),
            (404, StatusCodes.BadInvalidArgument),
            (422, StatusCodes.BadInvalidArgument),
            (409, StatusCodes.BadInvalidArgument),
            (503, StatusCodes.BadCommunicationError),
        ],
    )
    async def test_gateway_refusals_become_status_codes(
        self,
        as_operator: GatewayClient,
        gateway: tuple[str, GatewayScript],
        status: int,
        expected: int,
    ) -> None:
        _, script = gateway
        script.command_status = status
        script.command_body = {"detail": "no"}
        result = await MethodHandlers(as_operator).reset_emergency_stop(None)
        assert code(result) == expected

    async def test_a_reply_cut_short_is_a_communication_error(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript]
    ) -> None:
        _, script = gateway
        script.truncate_command_body = True
        result = await MethodHandlers(as_operator).reset_emergency_stop(None)
        assert code(result) == StatusCodes.BadCommunicationError

    async def test_an_expired_access_token_is_refreshed_once(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript]
    ) -> None:
        _, script = gateway
        script.reject_access_once = True
        result = await MethodHandlers(as_operator).reset_emergency_stop(None)
        assert outputs(result) == (True, "")
        assert [call.path for call in script.calls] == [
            "/auth/login",
            "/api/v1/commands/reset",
            "/auth/refresh",
            "/api/v1/commands/reset",
        ]
        assert script.calls[-1].authorization == "Bearer access-2"

    async def test_a_session_that_cannot_be_refreshed_is_denied(
        self, as_operator: GatewayClient, gateway: tuple[str, GatewayScript]
    ) -> None:
        _, script = gateway
        script.reject_access_once = True
        script.refresh_works = False
        result = await MethodHandlers(as_operator).reset_emergency_stop(None)
        assert code(result) == StatusCodes.BadUserAccessDenied

    async def test_a_wrong_password_is_denied(
        self, gateway: tuple[str, GatewayScript]
    ) -> None:
        url, _ = gateway
        client = GatewayClient(url)
        user = GatewayUserManager(client).get_user(None, "operator1", "wrong-password")
        token = CURRENT_USER.set(user)
        try:
            result = await MethodHandlers(client).reset_emergency_stop(None)
        finally:
            CURRENT_USER.reset(token)
        assert code(result) == StatusCodes.BadUserAccessDenied

    async def test_an_anonymous_session_cannot_call_methods(self) -> None:
        result = await MethodHandlers(GatewayClient("http://x")).reset_emergency_stop(
            None
        )
        assert code(result) == StatusCodes.BadUserAccessDenied

    async def test_an_unreachable_gateway_is_a_communication_error(self) -> None:
        client = GatewayClient("http://127.0.0.1:1")

        async def signed_in() -> GatewayTokens:
            return GatewayTokens(
                access_token="a",
                refresh_token="r",
                access_expires_at_ms=int(time.time() * 1000) + 900_000,
                username="operator1",
                role="operator",
            )

        login = asyncio.get_running_loop().create_task(signed_in())
        user = GatewayUser(
            role=UserRole.User, name="operator1", session=GatewaySession(client, login)
        )
        token = CURRENT_USER.set(user)
        try:
            result = await MethodHandlers(client).reset_emergency_stop(None)
        finally:
            CURRENT_USER.reset(token)
        assert code(result) == StatusCodes.BadCommunicationError


class FakePLC:
    def __init__(self) -> None:
        self.down = False

    async def get_control_status(self) -> pb.PLCStatusMsg:
        if self.down:
            raise grpc.RpcError("PLCService unreachable")
        return pb.PLCStatusMsg(mode=pb.ControlMode.MANUAL)


class FakeAlarms:
    def __init__(self) -> None:
        self.down = False
        self.calls = 0

    async def open_alarms(self, limit: int = 200) -> pb.AlarmListMsg:
        self.calls += 1
        if self.down:
            raise grpc.RpcError("AlarmService unreachable")
        return pb.AlarmListMsg(alarms=[pb.AlarmMsg(alarm_id=1, severity="warning")])


async def until(predicate: Any) -> None:
    async with asyncio.timeout(5.0):
        while not predicate():
            await asyncio.sleep(0.001)


class TestProjections:
    async def test_the_plc_folder_follows_the_plc_and_marks_an_outage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        opc, plc = RecordingOPC(), FakePLC()
        with caplog.at_level(logging.INFO, logger="opcua_server.upstreams"):
            task = asyncio.create_task(upstreams.run_plc_projection(opc, plc, 0.001))  # type: ignore[arg-type]
            try:
                await until(lambda: any(nid == 2700 for nid, _, _ in opc.updates))
                assert opc.latest(2700) == "manual"
                plc.down = True
                await until(lambda: opc.stale)
                assert opc.latest(2712) is False
                plc.down = False
                await until(lambda: opc.latest(2712) is True)
            finally:
                task.cancel()
        assert 2712 not in opc.stale
        assert caplog.text.count("PLCService unreachable for the PLC folder") == 1
        assert "PLCService reachable again" in caplog.text

    async def test_the_alarm_folder_refreshes_at_once_on_a_change(self) -> None:
        opc, alarms = RecordingOPC(), FakeAlarms()
        changed = asyncio.Event()
        task = asyncio.create_task(
            upstreams.run_alarm_projection(opc, alarms, 3600.0, changed)  # type: ignore[arg-type]
        )
        try:
            await until(lambda: alarms.calls == 1)
            changed.set()
            await until(lambda: alarms.calls == 2)
            assert opc.latest(2800) == 1
            alarms.down = True
            changed.set()
            await until(lambda: opc.stale)
            assert opc.latest(2804) is False
            alarms.down = False
            changed.set()
            await until(lambda: opc.latest(2804) is True)
        finally:
            task.cancel()


def plant_payload(**overrides: Any) -> bytes:
    values: dict[str, Any] = {
        "sensors": [
            pb.SensorStatusMsg(sensor_id="drum_level", quality=pb.SensorQuality.BAD)
        ],
        "timestamp_ms": 5_000,
    }
    values.update(overrides)
    return pb.PlantStatusMsg(**values).SerializeToString()


def boiler_payload(level: float) -> bytes:
    return pb.BoilerStateMsg(
        water_level_m=level, timestamp_ms=5_000
    ).SerializeToString()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def skipped(reason: str) -> float:
    value = REGISTRY.get_sample_value("opcua_bridge_skipped_total", {"reason": reason})
    return value or 0.0


class TestBridge:
    async def test_instrument_quality_follows_the_node(self) -> None:
        opc = RecordingOPC()
        bridge = MQTTOPCBridge(opc, max_update_hz=0)  # type: ignore[arg-type]
        await bridge._handle_message("sensors/plant", plant_payload())
        await bridge._handle_message("sensors/boiler", boiler_payload(4.5))
        level = [(v, q) for nid, v, q in opc.updates if nid == 2101]
        assert level == [(4.5, pb.SensorQuality.BAD)]

    async def test_a_burst_is_thinned_to_the_latest_values(self) -> None:
        opc, clock = RecordingOPC(), FakeClock()
        bridge = MQTTOPCBridge(opc, max_update_hz=5.0, clock=clock)  # type: ignore[arg-type]
        for level in (4.1, 4.2, 4.3):
            await bridge._handle_message("sensors/boiler", boiler_payload(level))
            clock.now += 0.05
        assert [v for nid, v, _ in opc.updates if nid == 2101] == [4.1]
        clock.now = 0.2
        task = asyncio.create_task(bridge.flush_pending())
        try:
            await until(
                lambda: len([1 for nid, _, _ in opc.updates if nid == 2101]) == 2
            )
        finally:
            task.cancel()
        assert opc.latest(2101) == 4.3
        assert bridge.stats == {"received": 3, "mapped": 2, "skipped": 0}

    async def test_a_message_within_the_interval_waits_for_its_turn(self) -> None:
        opc, clock = RecordingOPC(), FakeClock()
        bridge = MQTTOPCBridge(opc, max_update_hz=5.0, clock=clock)  # type: ignore[arg-type]
        await bridge._handle_message("sensors/boiler", boiler_payload(4.1))
        clock.now = 0.19
        await bridge._handle_message("sensors/boiler", boiler_payload(4.2))
        clock.now = 0.4
        await bridge._handle_message("sensors/boiler", boiler_payload(4.3))
        assert [v for nid, v, _ in opc.updates if nid == 2101] == [4.1, 4.3]

    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            "not bytes",
            None,
            pb.BoilerStateMsg(water_level_m=4.0).SerializeToString(),
        ],
    )
    async def test_an_empty_or_timeless_payload_writes_nothing(
        self, payload: object
    ) -> None:
        opc = RecordingOPC()
        bridge = MQTTOPCBridge(opc, max_update_hz=0)  # type: ignore[arg-type]
        await bridge._handle_message("sensors/boiler", payload)  # type: ignore[arg-type]
        assert opc.updates == []
        assert bridge.stats == {"received": 1, "mapped": 0, "skipped": 1}

    async def test_a_plant_status_without_a_timestamp_changes_no_quality(
        self,
    ) -> None:
        opc = RecordingOPC()
        bridge = MQTTOPCBridge(opc, max_update_hz=0)  # type: ignore[arg-type]
        await bridge._handle_message("sensors/plant", plant_payload(timestamp_ms=0))
        await bridge._handle_message("sensors/boiler", boiler_payload(4.5))
        level = [(v, q) for nid, v, q in opc.updates if nid == 2101]
        assert level == [(4.5, pb.SensorQuality.GOOD)]

    async def test_skipped_messages_are_counted_by_reason(self) -> None:
        bridge = MQTTOPCBridge(RecordingOPC(), max_update_hz=0)  # type: ignore[arg-type]
        before = {reason: skipped(reason) for reason in ("empty", "malformed")}
        await bridge._handle_message("sensors/boiler", b"")
        await bridge._handle_message("sensors/boiler", b"\xff\xff")
        assert skipped("empty") == before["empty"] + 1
        assert skipped("malformed") == before["malformed"] + 1

    async def test_a_defect_while_applying_is_logged_once_and_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class BrokenOPC(RecordingOPC):
            async def update_variable(self, node_id: int, value: Any, **_: Any) -> None:
                raise RuntimeError("defect in a projection")

        bridge = MQTTOPCBridge(BrokenOPC(), max_update_hz=0)  # type: ignore[arg-type]
        with caplog.at_level(logging.DEBUG, logger="opcua_server.subscriber"):
            await bridge._handle_message("sensors/boiler", boiler_payload(4.0))
            await bridge._handle_message("sensors/boiler", boiler_payload(4.1))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "MQTT" not in errors[0].getMessage()
        assert bridge.stats == {"received": 2, "mapped": 0, "skipped": 2}

    async def test_a_defect_in_a_held_message_does_not_stop_the_flush(self) -> None:
        class FlakyOPC(RecordingOPC):
            fail = True

            async def update_variable(
                self, node_id: int, value: Any, **kw: Any
            ) -> None:
                if self.fail:
                    raise RuntimeError("defect in a projection")
                await super().update_variable(node_id, value, **kw)

        opc, clock = FlakyOPC(), FakeClock()
        bridge = MQTTOPCBridge(opc, max_update_hz=5.0, clock=clock)  # type: ignore[arg-type]
        await bridge._handle_message("sensors/boiler", boiler_payload(4.1))
        await bridge._handle_message("sensors/boiler", boiler_payload(4.2))
        clock.now = 0.2
        task = asyncio.create_task(bridge.flush_pending())
        try:
            await until(lambda: bridge.stats["skipped"] == 2)
            opc.fail = False
            await bridge._handle_message("sensors/boiler", boiler_payload(4.3))
            clock.now = 0.4
            await until(lambda: any(nid == 2101 for nid, _, _ in opc.updates))
        finally:
            task.cancel()
        assert not task.done() or task.cancelled()
        assert opc.latest(2101) == 4.3

    async def test_an_alarm_change_wakes_the_alarm_folder(self) -> None:
        changed = asyncio.Event()
        bridge = MQTTOPCBridge(RecordingOPC(), alarms_changed=changed)  # type: ignore[arg-type]
        await bridge._handle_message("alarms/changes", b"{}")
        assert changed.is_set()

    async def test_an_unknown_node_is_logged_and_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        opc = RecordingOPC()
        opc.missing = {2101}
        bridge = MQTTOPCBridge(opc, max_update_hz=0)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger="opcua_server.subscriber"):
            await bridge._handle_message("sensors/boiler", boiler_payload(4.0))
        assert "OPC UA node 2101 not found" in caplog.text
        assert opc.latest(2100) == 0.0

    async def test_the_bridge_subscribes_and_reconnects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections: list[dict[str, Any]] = []
        subscriptions: list[tuple[str, int]] = []

        class Broker:
            def __init__(self, **options: Any) -> None:
                connections.append(options)

            async def __aenter__(self) -> Broker:
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            async def subscribe(self, topic: str, qos: int) -> None:
                subscriptions.append((topic, qos))

            @property
            def messages(self) -> AsyncIterator[SimpleNamespace]:
                async def deliver() -> AsyncIterator[SimpleNamespace]:
                    yield SimpleNamespace(topic="sensors/plant", payload="not bytes")
                    raise MqttError("connection lost")

                return deliver()

        monkeypatch.setattr(subscriber, "Client", Broker)
        monkeypatch.setattr(subscriber, "RECONNECT_DELAY_S", 0.001)
        opc = RecordingOPC()
        bridge = MQTTOPCBridge(
            opc,  # type: ignore[arg-type]
            mqtt_username="opcua-server",
            mqtt_password="pw",
        )
        task = asyncio.create_task(bridge.run())
        try:
            await until(lambda: len(connections) >= 2)
        finally:
            task.cancel()
        assert subscriptions[:2] == [("sensors/#", 0), ("alarms/changes", 1)]
        assert connections[0]["username"] == "opcua-server"
        assert bridge.stats["received"] >= 1
        assert opc.updates == []


class TestEntryPoint:
    def test_the_command_line_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["opcua_server"])
        args = entry.parse_args()
        assert (args.opc_port, args.metrics_port) == (4840, 9105)
        assert args.gateway_url == "http://localhost:8000"

    async def test_it_wires_the_server_bridge_and_projections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        class Server:
            def __init__(self, endpoint: str, **options: Any) -> None:
                events.append(f"server {endpoint} {options['gateway_url']}")

            async def start(self) -> None:
                events.append("started")

            async def stop(self) -> None:
                events.append("stopped")

        class Bridge:
            stats = {"received": 0, "mapped": 0, "skipped": 0}

            def __init__(self, opc: Any, **options: Any) -> None:
                events.append(f"bridge {options['mqtt_username']}")

            async def run(self) -> None:
                raise RuntimeError("broker config wrong")

            async def flush_pending(self) -> None:
                await asyncio.Event().wait()

        class Client:
            def __init__(self, target: str) -> None:
                events.append(f"client {target}")

            async def close(self) -> None:
                events.append("client closed")

        async def idle(*_: Any) -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(entry, "CogniBoilerOPCServer", Server)
        monkeypatch.setattr(entry, "MQTTOPCBridge", Bridge)
        monkeypatch.setattr(entry, "PLCStatusClient", Client)
        monkeypatch.setattr(entry, "AlarmReadClient", Client)
        monkeypatch.setattr(entry, "run_plc_projection", idle)
        monkeypatch.setattr(entry, "run_alarm_projection", idle)
        monkeypatch.setattr(entry, "start_metrics_server", lambda port, host: None)
        monkeypatch.delenv("MQTT_USERNAME", raising=False)
        monkeypatch.setattr("sys.argv", ["opcua_server", "--opc-port", "4999"])
        with pytest.raises(RuntimeError, match="broker config wrong"):
            await entry.main(entry.parse_args())
        assert (
            events[0]
            == "server opc.tcp://0.0.0.0:4999/cogniboiler http://localhost:8000"
        )
        assert "bridge opcua-server" in events
        assert events[-3:] == ["stopped", "client closed", "client closed"]


class TestReadClients:
    async def test_the_plc_status_and_open_alarms_are_read_over_grpc(self) -> None:
        requests: list[pb.ListAlarmsRequest] = []

        class PLC(pb2_grpc.PLCServiceServicer):
            async def GetControlStatus(self, request: Any, context: Any) -> Any:  # noqa: N802
                return pb.PLCStatusMsg(mode=pb.ControlMode.MANUAL)

        class Alarms(pb2_grpc.AlarmServiceServicer):
            async def ListAlarms(self, request: Any, context: Any) -> Any:  # noqa: N802
                requests.append(request)
                return pb.AlarmListMsg(total=3)

        server = grpc.aio.server()
        pb2_grpc.add_PLCServiceServicer_to_server(PLC(), server)
        pb2_grpc.add_AlarmServiceServicer_to_server(Alarms(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        plc = PLCStatusClient(f"127.0.0.1:{port}")
        alarms = AlarmReadClient(f"127.0.0.1:{port}")
        try:
            status = await plc.get_control_status()
            listed = await alarms.open_alarms(limit=50)
        finally:
            await plc.close()
            await alarms.close()
            await server.stop(grace=None)
        assert status.mode == pb.ControlMode.MANUAL
        assert listed.total == 3
        assert (requests[0].open_only, requests[0].limit) == (True, 50)
