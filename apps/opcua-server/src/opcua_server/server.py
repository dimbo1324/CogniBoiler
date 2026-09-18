"""
Async OPC UA server for the CogniBoiler digital twin.

Builds the address space of address_space.py under Objects/CogniBoiler (ns=2) and keeps
it current: the MQTT bridge writes plant values, the PLC and alarm projections write
their folders. Clients read and subscribe; they write only through methods, which run
as the session's user through the API gateway.

Usage:
    server = CogniBoilerOPCServer(gateway_url="http://localhost:8000")
    await server.start()
    await server.update_variable(NODEID_PRESSURE, 14_200_000.0)
    await server.stop()
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime

from asyncua import ua
from asyncua.common.node import Node
from asyncua.server.server import Server

from opcua_server.address_space import (
    ALL_VARIABLES,
    FOLDER_NODE_IDS,
    NAMESPACE_URI,
    NODEID_METHOD_ACKNOWLEDGE_ALARM,
    NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS,
    NODEID_METHOD_APPLY_VALVE_COMMAND,
    NODEID_METHOD_RESET_EMERGENCY_STOP,
    NODEID_METHOD_SET_CONTROL_MODE,
    NODEID_METHOD_SET_LOAD_DEMAND,
    NODEID_ROOT,
    NS_IDX,
    VARIABLES_BY_NODE_ID,
    Folder,
    InitialValue,
    ValueKind,
    VariableDescriptor,
)
from opcua_server.gateway import GatewayClient
from opcua_server.identity import install_identity
from opcua_server.methods import RESULT_ARGUMENTS, MethodHandlers, argument
from opcua_server.security import ServerCertificate, secure
from opcua_server.ua_types import (
    AttributeIds,
    ObjectIds,
    StatusCodes,
    node_id,
    status,
    timestamp,
)
from opcua_server.units import engineering_units

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT: str = "opc.tcp://0.0.0.0:4840/cogniboiler"
EU_PROPERTY_OFFSET = 100_000

QUALITY_GOOD = 0
QUALITY_UNCERTAIN = 1
QUALITY_BAD = 2

_STATUS_BY_QUALITY: dict[int, int] = {
    QUALITY_GOOD: StatusCodes.Good,
    QUALITY_UNCERTAIN: StatusCodes.UncertainSensorNotAccurate,
    QUALITY_BAD: StatusCodes.BadSensorFailure,
}

_VARIANT_TYPES: dict[ValueKind, ua.VariantType] = {
    ValueKind.DOUBLE: ua.VariantType.Double,
    ValueKind.BOOLEAN: ua.VariantType.Boolean,
    ValueKind.STRING: ua.VariantType.String,
    ValueKind.INT64: ua.VariantType.Int64,
    ValueKind.STRING_ARRAY: ua.VariantType.String,
}


def _coerce(kind: ValueKind, value: InitialValue) -> InitialValue:
    if kind is ValueKind.DOUBLE:
        return float(value)  # type: ignore[arg-type]
    if kind is ValueKind.BOOLEAN:
        return bool(value)
    if kind is ValueKind.INT64:
        return int(value)  # type: ignore[arg-type]
    if kind is ValueKind.STRING_ARRAY:
        return (
            [str(item) for item in value] if isinstance(value, list) else [str(value)]
        )
    return str(value)


def _localized(text: str) -> ua.DataValue:
    return ua.DataValue(
        ua.Variant(ua.LocalizedText(text), ua.VariantType.LocalizedText)
    )


class CogniBoilerOPCServer:
    """
    OPC UA server exposing CogniBoiler process variables, PLC state and alarms.

    Lifecycle:
        await server.start()   # builds address space, opens TCP port
        await server.stop()    # graceful shutdown
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        gateway_url: str = "http://localhost:8000",
        certificate: ServerCertificate | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._certificate = certificate
        self._gateway = GatewayClient(gateway_url)
        self._server = Server()
        self._ns: int = NS_IDX
        self._nodes: dict[int, Node] = {}
        self._started: bool = False
        self._methods = MethodHandlers(self._gateway)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Build the address space, install identity handling and open the port."""
        await self._server.init()
        self._server.set_endpoint(self._endpoint)
        self._server.set_server_name("CogniBoiler Digital Twin")
        self._server.set_identity_tokens(
            [ua.AnonymousIdentityToken, ua.UserNameIdentityToken]
        )
        install_identity(self._server, self._gateway)
        await secure(self._server, self._certificate)

        self._ns = await self._server.register_namespace(NAMESPACE_URI)
        root = await self._server.nodes.objects.add_folder(
            node_id(NODEID_ROOT, self._ns), "CogniBoiler"
        )
        folders: dict[Folder, Node] = {}
        for folder, folder_id in FOLDER_NODE_IDS.items():
            folders[folder] = await root.add_folder(
                node_id(folder_id, self._ns), folder.value
            )
        for descriptor in ALL_VARIABLES:
            self._nodes[descriptor.node_id] = await self._create_variable(
                folders[descriptor.folder], descriptor
            )
        await self._add_methods(folders[Folder.PLC], folders[Folder.ALARMS])

        await self._server.start()
        self._started = True
        logger.info(
            "OPC UA server started at %s with %d variables",
            self._endpoint,
            len(self._nodes),
        )

    async def stop(self) -> None:
        """Gracefully shut down the OPC UA server."""
        if self._started:
            await self._server.stop()
            self._started = False
            logger.info("OPC UA server stopped")

    # ─── Address space ────────────────────────────────────────────────────────

    async def _create_variable(
        self, parent: Node, descriptor: VariableDescriptor
    ) -> Node:
        """A read-only variable with display name, description and engineering units."""
        variant_type = _VARIANT_TYPES[descriptor.kind]
        node = await parent.add_variable(
            node_id(descriptor.node_id, self._ns),
            ua.QualifiedName(descriptor.browse_name, self._ns),
            _coerce(descriptor.kind, descriptor.initial_value),
            variant_type,
        )
        await node.write_attribute(
            AttributeIds.DisplayName, _localized(descriptor.display_name)
        )
        await node.write_attribute(
            AttributeIds.Description, _localized(descriptor.description)
        )
        units = engineering_units(descriptor.unit)
        if units is not None:
            await node.add_property(
                node_id(descriptor.node_id + EU_PROPERTY_OFFSET, self._ns),
                ua.QualifiedName("EngineeringUnits", 0),
                ua.Variant(units, ua.VariantType.ExtensionObject),
                datatype=node_id(ObjectIds.EUInformation),
            )
        return node

    async def _add_methods(self, plc: Node, alarms: Node) -> None:
        handlers = self._methods
        definitions: list[
            tuple[Node, int, str, Callable[..., Awaitable[object]], list[ua.Argument]]
        ] = [
            (
                plc,
                NODEID_METHOD_SET_LOAD_DEMAND,
                "SetLoadDemand",
                handlers.set_load_demand,
                [
                    argument(
                        "LoadW", ua.VariantType.Double, "Electrical load target [W]."
                    )
                ],
            ),
            (
                plc,
                NODEID_METHOD_SET_CONTROL_MODE,
                "SetControlMode",
                handlers.set_control_mode,
                [argument("Mode", ua.VariantType.String, "auto, manual or estop.")],
            ),
            (
                plc,
                NODEID_METHOD_RESET_EMERGENCY_STOP,
                "ResetEmergencyStop",
                handlers.reset_emergency_stop,
                [],
            ),
            (
                plc,
                NODEID_METHOD_APPLY_VALVE_COMMAND,
                "ApplyValveCommand",
                handlers.apply_valve_command,
                [
                    argument("FuelValve", ua.VariantType.Double, "Opening [0..1]."),
                    argument(
                        "FeedwaterValve", ua.VariantType.Double, "Opening [0..1]."
                    ),
                    argument("SteamValve", ua.VariantType.Double, "Opening [0..1]."),
                ],
            ),
            (
                alarms,
                NODEID_METHOD_ACKNOWLEDGE_ALARM,
                "AcknowledgeAlarm",
                handlers.acknowledge_alarm,
                [
                    argument("AlarmId", ua.VariantType.Int64, "Alarm to acknowledge."),
                    argument("Comment", ua.VariantType.String, "Up to 500 characters."),
                ],
            ),
            (
                alarms,
                NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS,
                "AcknowledgeAllAlarms",
                handlers.acknowledge_all_alarms,
                [argument("Comment", ua.VariantType.String, "Up to 500 characters.")],
            ),
        ]
        for parent, method_id, name, callback, inputs in definitions:
            await parent.add_method(
                node_id(method_id, self._ns),
                ua.QualifiedName(name, self._ns),
                callback,
                inputs,
                RESULT_ARGUMENTS,
            )

    # ─── Value updates ────────────────────────────────────────────────────────

    async def update_variable(
        self,
        node_id: int,
        value: InitialValue,
        *,
        quality: int = QUALITY_GOOD,
        source_timestamp_ms: int | None = None,
    ) -> None:
        """
        Write a new value to a variable node.

        Raises:
            KeyError: If node_id is not registered in the address space.
        """
        node = self._nodes[node_id]
        await self._write(
            node,
            VARIABLES_BY_NODE_ID[node_id],
            value,
            _STATUS_BY_QUALITY.get(quality, StatusCodes.Bad),
            source_timestamp_ms,
        )

    async def mark_stale(self, node_ids: Iterable[int]) -> None:
        """Keep the last values but flag them as no longer current."""
        for stale_id in node_ids:
            node = self._nodes[stale_id]
            value = await node.read_value()
            await self._write(
                node,
                VARIABLES_BY_NODE_ID[stale_id],
                value,
                StatusCodes.UncertainLastUsableValue,
                None,
            )

    async def _write(
        self,
        node: Node,
        descriptor: VariableDescriptor,
        value: InitialValue,
        code: int,
        source_timestamp_ms: int | None,
    ) -> None:
        now = datetime.now(UTC)
        source = (
            datetime.fromtimestamp(source_timestamp_ms / 1000, UTC)
            if source_timestamp_ms
            else now
        )
        data_value = ua.DataValue(
            Value=ua.Variant(
                _coerce(descriptor.kind, value), _VARIANT_TYPES[descriptor.kind]
            ),
            StatusCode=status(code),
            SourceTimestamp=timestamp(source),
            ServerTimestamp=timestamp(now),
        )
        await self._server.write_attribute_value(node.nodeid, data_value)

    def get_registered_node_ids(self) -> list[int]:
        """Return all node IDs currently registered in the address space."""
        return list(self._nodes.keys())

    @property
    def is_started(self) -> bool:
        return self._started
