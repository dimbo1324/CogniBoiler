"""
Tests for OPC UA address space definition and MQTT bridge logic.

No real OPC UA server needed — we test:
  1. address_space.py — node catalogue completeness and correctness
  2. MQTTOPCBridge._handle_message() — message parsing and routing
     using a mock OPC UA server
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from opcua_server.address_space import (
    ALL_VARIABLES,
    BOILER_VARIABLES,
    MQTT_TOPIC_TO_NODEID,
    NODEID_ELECTRICAL_POWER,
    NODEID_PRESSURE,
    NS_IDX,
    TURBINE_VARIABLES,
)
from opcua_server.subscriber import MQTTOPCBridge

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_payload(value: float, unit: str = "Pa") -> bytes:
    """Build a JSON sensor payload as the MQTT publisher would."""
    return json.dumps(
        {"value": value, "unit": unit, "quality": "good", "ts": 1}
    ).encode()


def make_mock_opc() -> MagicMock:
    """Mock OPC UA server with async update_variable."""
    opc = MagicMock()
    opc.update_variable = AsyncMock()
    return opc


# ─── Address space catalogue tests ───────────────────────────────────────────


class TestAddressSpace:
    def test_all_variables_have_unique_node_ids(self) -> None:
        ids = [v.node_id for v in ALL_VARIABLES]
        assert len(ids) == len(set(ids)), "Duplicate node IDs found"

    def test_all_variables_have_unique_browse_names(self) -> None:
        names = [v.browse_name for v in ALL_VARIABLES]
        assert len(names) == len(set(names)), "Duplicate browse names found"

    def test_boiler_variables_count(self) -> None:
        assert len(BOILER_VARIABLES) == 5

    def test_turbine_variables_count(self) -> None:
        assert len(TURBINE_VARIABLES) == 4

    def test_total_variables_count(self) -> None:
        assert len(ALL_VARIABLES) == 9

    def test_all_variables_have_non_empty_unit(self) -> None:
        for var in ALL_VARIABLES:
            assert var.unit != "", f"Empty unit for {var.browse_name}"

    def test_all_variables_have_non_empty_description(self) -> None:
        for var in ALL_VARIABLES:
            assert var.description != "", f"Empty description for {var.browse_name}"

    def test_pressure_node_id_correct(self) -> None:
        assert NODEID_PRESSURE == 2100

    def test_electrical_power_node_id_correct(self) -> None:
        assert NODEID_ELECTRICAL_POWER == 2200

    def test_pressure_initial_value_reasonable(self) -> None:
        pressure_var = next(v for v in BOILER_VARIABLES if v.browse_name == "Pressure")
        # 50 bar < nominal < 200 bar
        assert 50.0e5 < pressure_var.initial_value < 200.0e5

    def test_ns_idx_is_two(self) -> None:
        """Namespace 0 and 1 are reserved by OPC UA spec."""
        assert NS_IDX == 2

    def test_mqtt_mapping_covers_all_variables(self) -> None:
        """Every variable must have an MQTT topic mapping."""
        mapped_node_ids = set(MQTT_TOPIC_TO_NODEID.values())
        for var in ALL_VARIABLES:
            assert var.node_id in mapped_node_ids, (
                f"Variable {var.browse_name} (id={var.node_id}) "
                f"has no MQTT topic mapping"
            )

    def test_mqtt_mapping_no_duplicate_node_ids(self) -> None:
        node_ids = list(MQTT_TOPIC_TO_NODEID.values())
        assert len(node_ids) == len(set(node_ids))

    def test_all_mqtt_topics_start_with_sensors(self) -> None:
        for topic in MQTT_TOPIC_TO_NODEID:
            assert topic.startswith("sensors/")


# ─── MQTT→OPC bridge tests ────────────────────────────────────────────────────


class TestMQTTOPCBridge:
    @pytest.fixture
    def opc(self) -> MagicMock:
        return make_mock_opc()

    @pytest.fixture
    def bridge(self, opc: MagicMock) -> MQTTOPCBridge:
        return MQTTOPCBridge(opc_server=opc)

    @pytest.mark.asyncio
    async def test_known_topic_calls_update_variable(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        payload = make_payload(14_000_000.0)
        await bridge._handle_message("sensors/boiler/pressure_pa", payload)
        opc.update_variable.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_known_topic_passes_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        payload = make_payload(14_000_000.0)
        await bridge._handle_message("sensors/boiler/pressure_pa", payload)
        call_args = opc.update_variable.call_args
        assert call_args.args[0] == NODEID_PRESSURE

    @pytest.mark.asyncio
    async def test_known_topic_passes_correct_value(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        payload = make_payload(14_200_000.0)
        await bridge._handle_message("sensors/boiler/pressure_pa", payload)
        call_args = opc.update_variable.call_args
        assert call_args.args[1] == pytest.approx(14_200_000.0)

    @pytest.mark.asyncio
    async def test_heartbeat_topic_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/system/timestamp_ms", b"1741000000000")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_topic_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/unknown/variable", make_payload(1.0))
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_json_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/boiler/pressure_pa", b"not-json")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_value_key_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        payload = json.dumps({"unit": "Pa", "quality": "good"}).encode()
        await bridge._handle_message("sensors/boiler/pressure_pa", payload)
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_received_counter_increments(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/boiler/pressure_pa", make_payload(1.0))
        assert bridge.stats["received"] == 1

    @pytest.mark.asyncio
    async def test_mapped_counter_increments_on_success(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/boiler/pressure_pa", make_payload(1.0))
        assert bridge.stats["mapped"] == 1

    @pytest.mark.asyncio
    async def test_skipped_counter_increments_on_unknown_topic(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/unknown/xyz", make_payload(1.0))
        assert bridge.stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_skipped_counter_increments_on_heartbeat(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/system/timestamp_ms", b"12345")
        assert bridge.stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_all_sensor_topics_are_mapped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        """Every MQTT sensor topic must successfully route to OPC UA."""
        for topic in MQTT_TOPIC_TO_NODEID:
            opc.update_variable.reset_mock()
            await bridge._handle_message(topic, make_payload(1.0))
            opc.update_variable.assert_awaited_once()
