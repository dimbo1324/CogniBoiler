"""The OPC UA server end to end: a real asyncua server, client and a stand-in gateway."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from asyncua import Client, ua
from asyncua.common.utils import ServiceError
from asyncua.server.internal_session import InternalSession
from opcua_fakes import PASSWORD, GatewayScript, gateway_server
from opcua_server.address_space import (
    ALL_VARIABLES,
    NODEID_ALARMS_FOLDER,
    NODEID_ELECTRICAL_POWER,
    NODEID_METHOD_ACKNOWLEDGE_ALARM,
    NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS,
    NODEID_METHOD_SET_LOAD_DEMAND,
    NODEID_PLC_FOLDER,
    NODEID_PRESSURE,
    NODEID_ROOT,
    NODEID_RUN_ID,
    NS_IDX,
    VARIABLES_BY_NODE_ID,
)
from opcua_server.identity import drain_sign_outs
from opcua_server.security import (
    APPLICATION_URI,
    ServerCertificate,
    self_signed_certificate,
)
from opcua_server.server import (
    EU_PROPERTY_OFFSET,
    MAX_CHUNKS,
    MAX_CONNECTIONS,
    MAX_MESSAGE_BYTES,
    MAX_SESSIONS,
    MAX_SUBSCRIPTIONS,
    QUALITY_BAD,
    QUALITY_UNCERTAIN,
    TRANSPORT_BUFFER_BYTES,
    CogniBoilerOPCServer,
    endpoint_url,
)
from opcua_server.ua_types import StatusCodes
from opcua_server.units import engineering_units


@cache
def server_certificate() -> ServerCertificate:
    return self_signed_certificate()


@pytest.fixture
def gateway() -> Iterator[tuple[str, GatewayScript]]:
    with gateway_server() as found:
        yield found


@pytest_asyncio.fixture
async def opc(
    gateway: tuple[str, GatewayScript],
) -> AsyncIterator[tuple[CogniBoilerOPCServer, str]]:
    url, _ = gateway
    server = CogniBoilerOPCServer(
        endpoint_url("127.0.0.1", 0), gateway_url=url, certificate=server_certificate()
    )
    await server.start()
    try:
        yield server, server.bound_endpoint
    finally:
        await server.stop()


# asyncua enforces the chunk count on receive; this is one byte past what it admits.
MAX_REQUEST_BYTES = MAX_CHUNKS * TRANSPORT_BUFFER_BYTES + 1


def node(client: Client, identifier: int) -> Any:
    return client.get_node(ua.NodeId(identifier, NS_IDX))


async def settled(server: CogniBoilerOPCServer) -> None:
    """Every client connection is gone, its cleanup ran and its sign-out finished."""
    transport = server._server.bserver
    assert transport is not None
    async with asyncio.timeout(5.0):
        while transport.clients:
            await asyncio.sleep(0.01)
    await asyncio.gather(*transport.closing_tasks, return_exceptions=True)
    await drain_sign_outs(5.0)


class TestReading:
    async def test_an_anonymous_client_browses_and_reads_the_plant(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, endpoint = opc
        assert server.is_started
        assert NODEID_PRESSURE in server.get_registered_node_ids()
        async with Client(url=endpoint) as client:
            root = node(client, NODEID_ROOT)
            names = {
                (await child.read_browse_name()).Name
                for child in await root.get_children()
            }
            pressure = await node(client, NODEID_PRESSURE).read_value()
            units = await node(
                client, NODEID_PRESSURE + EU_PROPERTY_OFFSET
            ).read_value()
            display = await node(client, NODEID_PRESSURE).read_display_name()
        assert {"Boiler", "Turbine", "PLC", "Alarms"} <= names
        assert pressure == pytest.approx(140.0e5)
        assert units.DisplayName.Text == "Pa"
        assert display.Text == "Drum Pressure"

    async def test_updates_carry_quality_and_source_time(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, endpoint = opc
        await server.update_variable(
            NODEID_PRESSURE,
            150.0e5,
            quality=QUALITY_UNCERTAIN,
            source_timestamp_ms=1_741_000_000_000,
        )
        await server.update_variable(NODEID_ELECTRICAL_POWER, 1, quality=QUALITY_BAD)
        async with Client(url=endpoint) as client:
            pressure = await node(client, NODEID_PRESSURE).read_data_value(
                raise_on_bad_status=False
            )
            power = await node(client, NODEID_ELECTRICAL_POWER).read_data_value(
                raise_on_bad_status=False
            )
        assert pressure.Value.Value == pytest.approx(150.0e5)
        assert pressure.StatusCode.value == StatusCodes.UncertainSensorNotAccurate
        assert int(pressure.SourceTimestamp.timestamp()) == 1_741_000_000
        # A Bad status travels without its value (OPC UA Part 4).
        assert power.Value.Value is None
        assert power.StatusCode.value == StatusCodes.BadSensorFailure

    async def test_stale_values_are_kept_but_flagged(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, endpoint = opc
        await server.update_variable(NODEID_PRESSURE, 151.0e5)
        await server.mark_stale([NODEID_PRESSURE])
        async with Client(url=endpoint) as client:
            value = await node(client, NODEID_PRESSURE).read_data_value(
                raise_on_bad_status=False
            )
        assert value.Value.Value == pytest.approx(151.0e5)
        assert value.StatusCode.value == StatusCodes.UncertainLastUsableValue

    async def test_an_unknown_node_cannot_be_updated(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, _ = opc
        with pytest.raises(KeyError):
            await server.update_variable(123_456, 1.0)

    async def test_a_value_the_node_cannot_hold_is_reported_once(
        self, opc: tuple[CogniBoilerOPCServer, str], caplog: pytest.LogCaptureFixture
    ) -> None:
        server, endpoint = opc
        with caplog.at_level(logging.WARNING, logger="opcua_server.server"):
            await server.update_variable(NODEID_RUN_ID, ["not", "an", "integer"])
            await server.update_variable(NODEID_RUN_ID, ["again"])
        assert caplog.text.count("RunId") == 1
        await server.update_variable(NODEID_RUN_ID, 7)
        async with Client(url=endpoint) as client:
            assert await node(client, NODEID_RUN_ID).read_value() == 7
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="opcua_server.server"):
            await server.update_variable(NODEID_RUN_ID, ["after a good write"])
        assert caplog.text.count("RunId") == 1

    async def test_a_write_the_address_space_refuses_is_reported_once(
        self,
        opc: tuple[CogniBoilerOPCServer, str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        server, _ = opc
        aspace = server._server.iserver.aspace

        async def refuse(*_: Any) -> ua.StatusCode:
            return ua.StatusCode(StatusCodes.BadTypeMismatch)

        monkeypatch.setattr(aspace, "write_attribute_value", refuse)
        with caplog.at_level(logging.WARNING, logger="opcua_server.server"):
            await server.update_variable(NODEID_PRESSURE, 1.0)
            await server.update_variable(NODEID_PRESSURE, 2.0)
        assert caplog.text.count("BadTypeMismatch") == 1

    async def test_values_are_not_writable_by_clients(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        _, endpoint = opc
        async with Client(url=endpoint) as client:
            with pytest.raises(ua.UaStatusCodeError):
                await node(client, NODEID_PRESSURE).write_value(1.0)

    async def test_no_variable_or_property_is_writable(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        _, endpoint = opc
        identifiers = [
            *VARIABLES_BY_NODE_ID,
            *(
                descriptor.node_id + EU_PROPERTY_OFFSET
                for descriptor in ALL_VARIABLES
                if engineering_units(descriptor.unit) is not None
            ),
        ]
        async with Client(url=endpoint) as client:
            nodes = [node(client, identifier) for identifier in identifiers]
            values = await client.read_attributes(nodes)
            levels = [
                *await client.read_attributes(nodes, ua.AttributeIds.AccessLevel),
                *await client.read_attributes(nodes, ua.AttributeIds.UserAccessLevel),
            ]
        assert len(identifiers) > len(VARIABLES_BY_NODE_ID)
        assert all(value.StatusCode.is_good() for value in values)
        read = ua.AccessLevel.CurrentRead.mask
        write = ua.AccessLevel.CurrentWrite.mask
        assert all(level.Value.Value & read for level in levels)
        assert not [level for level in levels if level.Value.Value & write]

    @pytest.mark.parametrize("signed_in", [False, True])
    async def test_a_write_is_refused_for_anonymous_and_signed_in_users(
        self, opc: tuple[CogniBoilerOPCServer, str], signed_in: bool
    ) -> None:
        _, endpoint = opc
        client = Client(url=endpoint)
        if signed_in:
            client.set_user("operator1")
            client.set_password(PASSWORD)
        async with client:
            pressure = node(client, NODEID_PRESSURE)
            before = await pressure.read_value()
            with pytest.raises(ua.UaStatusCodeError) as refused:
                await pressure.write_value(ua.Variant(1.0, ua.VariantType.Double))
            with pytest.raises(ua.UaStatusCodeError):
                await node(client, NODEID_ROOT).add_variable(
                    ua.NodeId(99_999, NS_IDX), "Injected", 1.0
                )
            with pytest.raises(ua.UaStatusCodeError) as not_deleted:
                await client.delete_nodes([pressure])
            after = await pressure.read_value()
        assert refused.value.code == StatusCodes.BadUserAccessDenied
        assert not_deleted.value.code == StatusCodes.BadUserAccessDenied
        assert after == before


class TestMethods:
    async def test_an_anonymous_session_cannot_call_a_method(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        _, endpoint = opc
        _, script = gateway
        async with Client(url=endpoint) as client:
            with pytest.raises(ua.UaStatusCodeError) as refused:
                await node(client, NODEID_PLC_FOLDER).call_method(
                    ua.NodeId(NODEID_METHOD_SET_LOAD_DEMAND, NS_IDX),
                    ua.Variant(180e6, ua.VariantType.Double),
                )
        assert refused.value.code == StatusCodes.BadUserAccessDenied
        assert script.calls == []

    async def test_a_signed_in_user_commands_through_the_gateway(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        _, endpoint = opc
        _, script = gateway
        client = Client(url=endpoint)
        client.set_user("operator1")
        client.set_password(PASSWORD)
        async with client:
            outputs = await node(client, NODEID_PLC_FOLDER).call_method(
                ua.NodeId(NODEID_METHOD_SET_LOAD_DEMAND, NS_IDX),
                ua.Variant(180e6, ua.VariantType.Double),
            )
            ack = await node(client, NODEID_ALARMS_FOLDER).call_method(
                ua.NodeId(NODEID_METHOD_ACKNOWLEDGE_ALARM, NS_IDX),
                ua.Variant(7, ua.VariantType.Int64),
                ua.Variant("seen", ua.VariantType.String),
            )
        assert outputs == [True, ""]
        assert ack == [True, ""]
        paths = [call.path for call in script.calls]
        assert paths[:3] == [
            "/auth/login",
            "/api/v1/commands/load",
            "/api/v1/alarms/7/ack",
        ]
        assert script.calls[0].body == {"username": "operator1", "password": PASSWORD}
        assert script.calls[1].body == {"load_w": 180e6}
        assert {call.forwarded_for for call in script.calls} == {"127.0.0.1"}
        await settled(opc[0])
        logouts = [call for call in script.calls if call.path == "/auth/logout"]
        assert [call.body for call in logouts] == [{"refresh_token": "refresh-1"}]

    async def test_each_re_activation_signs_out_the_session_it_replaces(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        server, endpoint = opc
        _, script = gateway
        client = Client(url=endpoint)
        client.set_user("operator1")
        client.set_password(PASSWORD)
        async with client:
            await client.activate_session(username="operator1", password=PASSWORD)
            await client.activate_session(username="operator1", password=PASSWORD)
        await settled(server)
        logins = [call for call in script.calls if call.path == "/auth/login"]
        signed_out = [
            call.body["refresh_token"]
            for call in script.calls
            if call.path == "/auth/logout"
        ]
        assert len(logins) == 3
        assert sorted(signed_out) == ["refresh-1", "refresh-2", "refresh-3"]

    async def test_a_refusal_of_the_plc_is_an_output_not_an_error(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        _, endpoint = opc
        _, script = gateway
        script.command_body = {"accepted": False, "reason": "E-Stop active"}
        client = Client(url=endpoint)
        client.set_user("operator1")
        client.set_password(PASSWORD)
        async with client:
            outputs = await node(client, NODEID_PLC_FOLDER).call_method(
                ua.NodeId(NODEID_METHOD_SET_LOAD_DEMAND, NS_IDX),
                ua.Variant(1e6, ua.VariantType.Double),
            )
        assert outputs == [False, "E-Stop active"]


class TestEncryptedChannel:
    async def test_sign_and_encrypt_with_a_client_certificate(
        self, opc: tuple[CogniBoilerOPCServer, str], tmp_path: Path
    ) -> None:
        _, endpoint = opc
        own = self_signed_certificate(application_uri="urn:cogniboiler:test-client")
        cert, key = tmp_path / "client.pem", tmp_path / "client-key.pem"
        cert.write_bytes(own.certificate_pem)
        key.write_bytes(own.private_key_pem)
        client = Client(url=endpoint)
        client.application_uri = "urn:cogniboiler:test-client"
        await client.set_security_string(
            f"Basic256Sha256,SignAndEncrypt,{cert.as_posix()},{key.as_posix()}"
        )
        async with client:
            value = await node(client, NODEID_PRESSURE).read_value()
            server_uri = client.security_policy.peer_certificate is not None
        assert value == pytest.approx(140.0e5)
        assert server_uri

    async def test_the_server_names_its_application(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        _, endpoint = opc
        async with Client(url=endpoint) as client:
            endpoints = await client.get_endpoints()
        modes = {item.SecurityMode for item in endpoints}
        assert modes == {
            ua.MessageSecurityMode.None_,
            ua.MessageSecurityMode.SignAndEncrypt,
        }
        assert {item.Server.ApplicationUri for item in endpoints} == {APPLICATION_URI}


class TestLimits:
    async def test_the_server_revises_the_transport_and_session_limits(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, _ = opc
        limits = server._server.limits
        iserver = server._server.iserver
        assert limits.max_message_size == MAX_MESSAGE_BYTES
        assert limits.max_chunk_count * limits.max_recv_buffer >= MAX_MESSAGE_BYTES
        assert (iserver.max_connections, iserver.max_subscriptions) == (
            MAX_CONNECTIONS,
            MAX_SUBSCRIPTIONS,
        )
        assert InternalSession.max_connections == MAX_SESSIONS
        assert MAX_MESSAGE_BYTES < 100 * 1024 * 1024

    async def test_a_request_over_the_message_limit_is_refused(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        _, endpoint = opc
        _, script = gateway
        client = Client(url=endpoint)
        client.set_user("operator1")
        client.set_password(PASSWORD)
        async with client:
            accepted = await node(client, NODEID_ALARMS_FOLDER).call_method(
                ua.NodeId(NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS, NS_IDX),
                ua.Variant("x" * 1000, ua.VariantType.String),
            )
            with pytest.raises(ua.UaStatusCodeError) as refused:
                await node(client, NODEID_ALARMS_FOLDER).call_method(
                    ua.NodeId(NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS, NS_IDX),
                    ua.Variant("x" * MAX_REQUEST_BYTES, ua.VariantType.String),
                )
        assert accepted == [True, ""]
        assert refused.value.code == StatusCodes.BadRequestTooLarge
        commands = [c for c in script.calls if not c.path.startswith("/auth/")]
        assert len(commands) == 1


class TestInstalledIdentity:
    async def test_the_server_builds_user_aware_sessions_with_random_tokens(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        server, _ = opc
        iserver = server._server.iserver
        first = iserver.create_session(("127.0.0.1", 1), external=True)
        second = iserver.create_session(("127.0.0.1", 2), external=True)
        try:
            assert type(first).__name__ == "_UserAwareSession"
            assert first.auth_token.NodeIdType == ua.NodeIdType.ByteString
            assert first.auth_token != second.auth_token
            assert iserver.lookup_external_session(first.auth_token) is first
        finally:
            await first.close_session()
            await second.close_session()

    async def test_a_password_in_clear_on_the_open_endpoint_is_refused(
        self, opc: tuple[CogniBoilerOPCServer, str], gateway: tuple[str, GatewayScript]
    ) -> None:
        server, _ = opc
        _, script = gateway
        session = server._server.iserver.create_session(("127.0.0.1", 1), external=True)
        params = ua.ActivateSessionParameters()
        params.UserIdentityToken = ua.UserNameIdentityToken(
            UserName="operator1", Password=PASSWORD.encode(), EncryptionAlgorithm=None
        )
        try:
            with pytest.raises(ServiceError) as refused:
                session.activate_session(params, None)
            assert refused.value.code == StatusCodes.BadIdentityTokenRejected
            assert not session.is_activated()
            session.activate_session(params, b"client certificate")
            assert session.is_activated()
        finally:
            await session.close_session()
            await drain_sign_outs(5.0)
        logins = [call for call in script.calls if call.path == "/auth/login"]
        assert len(logins) == 1
