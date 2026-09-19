"""The OPC UA server end to end: a real asyncua server, client and a stand-in gateway."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from asyncua import Client, ua
from opcua_fakes import PASSWORD, GatewayScript, gateway_server
from opcua_server.address_space import (
    NODEID_ALARMS_FOLDER,
    NODEID_ELECTRICAL_POWER,
    NODEID_METHOD_ACKNOWLEDGE_ALARM,
    NODEID_METHOD_SET_LOAD_DEMAND,
    NODEID_PLC_FOLDER,
    NODEID_PRESSURE,
    NODEID_ROOT,
    NS_IDX,
)
from opcua_server.security import (
    APPLICATION_URI,
    ServerCertificate,
    self_signed_certificate,
)
from opcua_server.server import (
    EU_PROPERTY_OFFSET,
    QUALITY_BAD,
    QUALITY_UNCERTAIN,
    CogniBoilerOPCServer,
)
from opcua_server.ua_types import StatusCodes


@cache
def server_certificate() -> ServerCertificate:
    return self_signed_certificate()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


@pytest.fixture
def gateway() -> Iterator[tuple[str, GatewayScript]]:
    with gateway_server() as found:
        yield found


@pytest_asyncio.fixture
async def opc(
    gateway: tuple[str, GatewayScript],
) -> AsyncIterator[tuple[CogniBoilerOPCServer, str]]:
    url, _ = gateway
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/cogniboiler"
    server = CogniBoilerOPCServer(
        endpoint, gateway_url=url, certificate=server_certificate()
    )
    await server.start()
    try:
        yield server, endpoint
    finally:
        await server.stop()


def node(client: Client, identifier: int) -> Any:
    return client.get_node(ua.NodeId(identifier, NS_IDX))


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

    async def test_values_are_not_writable_by_clients(
        self, opc: tuple[CogniBoilerOPCServer, str]
    ) -> None:
        _, endpoint = opc
        async with Client(url=endpoint) as client:
            with pytest.raises(ua.UaStatusCodeError):
                await node(client, NODEID_PRESSURE).write_value(1.0)


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
        async with asyncio.timeout(5.0):
            while "/auth/logout" not in [call.path for call in script.calls]:
                await asyncio.sleep(0.02)

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
