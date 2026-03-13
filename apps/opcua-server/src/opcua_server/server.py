"""
Async OPC UA server for CogniBoiler digital twin.

Exposes boiler and turbine process variables as OPC UA VariableNodes
under the CogniBoiler namespace (ns=2).

The server does NOT pull data itself — it exposes an update_variable()
method that the MQTT subscriber calls whenever a new sensor reading arrives.

Architecture:
    MQTT subscriber → calls update_variable(node_id, value)
                    → OPC UA server writes new DataValue to the node
                    → OPC UA clients see the updated value via subscriptions

Usage:
    server = CogniBoilerOPCServer()
    await server.start()
    await server.update_variable(NODEID_PRESSURE, 14_200_000.0)
    ...
    await server.stop()
"""

from __future__ import annotations

import logging

from asyncua import Server, ua
from asyncua.common.node import Node

from opcua_server.address_space import (
    BOILER_VARIABLES,
    NAMESPACE_URI,
    NODEID_BOILER_FOLDER,
    NODEID_ROOT,
    NODEID_TURBINE_FOLDER,
    NS_IDX,
    TURBINE_VARIABLES,
    VariableDescriptor,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT: str = "opc.tcp://0.0.0.0:4840/cogniboiler"


class CogniBoilerOPCServer:
    """
    OPC UA server exposing CogniBoiler process variables.

    Lifecycle:
        await server.start()   # builds address space, opens TCP port
        await server.stop()    # graceful shutdown
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT) -> None:
        self._endpoint = endpoint
        self._server = Server()
        self._ns: int = NS_IDX
        self._nodes: dict[int, Node] = {}  # node_id → asyncua Node
        self._started: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Initialise the OPC UA server and build the address space.

        1. Register namespace URI → get namespace index
        2. Create folder hierarchy under Objects/
        3. Create all VariableNodes with initial values
        4. Start TCP listener
        """
        await self._server.init()
        self._server.set_endpoint(self._endpoint)
        self._server.set_server_name("CogniBoiler Digital Twin")

        # Register our namespace
        self._ns = await self._server.register_namespace(NAMESPACE_URI)

        # Build address space
        objects = self._server.nodes.objects
        root_folder = await objects.add_folder(
            ua.NodeId(NODEID_ROOT, self._ns), "CogniBoiler"
        )
        boiler_folder = await root_folder.add_folder(
            ua.NodeId(NODEID_BOILER_FOLDER, self._ns), "Boiler"
        )
        turbine_folder = await root_folder.add_folder(
            ua.NodeId(NODEID_TURBINE_FOLDER, self._ns), "Turbine"
        )

        # Create boiler variable nodes
        for desc in BOILER_VARIABLES:
            node = await self._create_variable(boiler_folder, desc)
            self._nodes[desc.node_id] = node

        # Create turbine variable nodes
        for desc in TURBINE_VARIABLES:
            node = await self._create_variable(turbine_folder, desc)
            self._nodes[desc.node_id] = node

        await self._server.start()
        self._started = True
        logger.info("OPC UA server started at %s", self._endpoint)

    async def stop(self) -> None:
        """Gracefully shut down the OPC UA server."""
        if self._started:
            await self._server.stop()
            self._started = False
            logger.info("OPC UA server stopped")

    # ─── Variable creation ────────────────────────────────────────────────────

    async def _create_variable(
        self,
        parent: Node,
        desc: VariableDescriptor,
    ) -> Node:
        """
        Create a writable VariableNode under parent folder.

        Sets:
          - NodeId, BrowseName, DisplayName
          - Initial value (Double)
          - Writable (so the MQTT subscriber can update it)
        """
        node = await parent.add_variable(
            ua.NodeId(desc.node_id, self._ns),
            desc.browse_name,
            desc.initial_value,
        )
        await node.set_writable()
        logger.debug(
            "Created OPC UA node ns=%d;i=%d  %s [%s]",
            self._ns,
            desc.node_id,
            desc.browse_name,
            desc.unit,
        )
        return node

    # ─── Value updates ────────────────────────────────────────────────────────

    async def update_variable(self, node_id: int, value: float) -> None:
        """
        Write a new value to an OPC UA variable node.

        Called by the MQTT subscriber when a new sensor reading arrives.

        Args:
            node_id: Integer node ID (e.g. NODEID_PRESSURE = 2100).
            value:   New engineering value.

        Raises:
            KeyError: If node_id is not registered in the address space.
        """
        node = self._nodes[node_id]
        await node.write_value(value)

    def get_registered_node_ids(self) -> list[int]:
        """Return all node IDs currently registered in the address space."""
        return list(self._nodes.keys())

    @property
    def is_started(self) -> bool:
        return self._started
