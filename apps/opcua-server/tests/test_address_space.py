"""
Tests for OPC UA address space definition and MQTT bridge logic.

No real OPC UA server needed — we test:
  1. address_space.py — node catalogue completeness and correctness
  2. MQTTOPCBridge._handle_message() — protobuf parsing and routing
     using a mock OPC UA server
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import cogniboiler_pb2 as pb
import pytest
from opcua_server.address_space import (
    ALL_VARIABLES,
    BOILER_FIELD_TO_NODEID,
    BOILER_VARIABLES,
    NODEID_ELECTRICAL_POWER,
    NODEID_EXHAUST_PRESSURE,
    NODEID_PRESSURE,
    NODEID_SHAFT_POWER,
    NODEID_STEAM_FLOW,
    NS_IDX,
    TURBINE_FIELD_TO_NODEID,
    TURBINE_VARIABLES,
)
from opcua_server.subscriber import (
    TOPIC_BOILER,
    TOPIC_HEARTBEAT,
    TOPIC_TURBINE,
    MQTTOPCBridge,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

TS_MS: int = 1_741_000_000_000


def make_boiler_payload(pressure_pa: float = 14_000_000.0) -> bytes:
    """Serialize a BoilerStateMsg as the publisher would."""
    return cast(
        bytes,
        pb.BoilerStateMsg(
            pressure_pa=pressure_pa,
            water_level_m=4.8,
            water_temp_k=611.0,
            flue_gas_temp_k=1200.0,
            internal_energy_j=2.5e12,
            quality=pb.SensorQuality.GOOD,
            timestamp_ms=TS_MS,
        ).SerializeToString(),
    )


def make_turbine_payload(electrical_power_w: float = 200_000_000.0) -> bytes:
    """Serialize a TurbineStateMsg as the publisher would."""
    return cast(
        bytes,
        pb.TurbineStateMsg(
            electrical_power_w=electrical_power_w,
            shaft_power_w=205_000_000.0,
            enthalpy_in_j_kg=3_400_000.0,
            enthalpy_out_j_kg=2_200_000.0,
            exhaust_pressure_pa=7_000.0,
            steam_flow_kg_s=150.0,
            timestamp_ms=TS_MS,
        ).SerializeToString(),
    )


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

    # ─── BOILER_FIELD_TO_NODEID tests ────────────────────────────────────────

    def test_boiler_field_mapping_covers_all_boiler_variables(self) -> None:
        """Every boiler variable must have a protobuf field mapping."""
        mapped_node_ids = set(BOILER_FIELD_TO_NODEID.values())
        for var in BOILER_VARIABLES:
            assert var.node_id in mapped_node_ids, (
                f"Boiler variable {var.browse_name} (id={var.node_id}) "
                f"has no protobuf field mapping"
            )

    def test_boiler_field_mapping_no_duplicate_node_ids(self) -> None:
        node_ids = list(BOILER_FIELD_TO_NODEID.values())
        assert len(node_ids) == len(set(node_ids))

    def test_boiler_field_mapping_count(self) -> None:
        assert len(BOILER_FIELD_TO_NODEID) == 5

    def test_boiler_field_keys_are_valid_proto_fields(self) -> None:
        """All keys must be real BoilerStateMsg field names."""
        dummy = pb.BoilerStateMsg()
        for field in BOILER_FIELD_TO_NODEID:
            assert hasattr(dummy, field), f"'{field}' is not a BoilerStateMsg field"

    # ─── TURBINE_FIELD_TO_NODEID tests ───────────────────────────────────────

    def test_turbine_field_mapping_covers_all_turbine_variables(self) -> None:
        """Every turbine variable must have a protobuf field mapping."""
        mapped_node_ids = set(TURBINE_FIELD_TO_NODEID.values())
        for var in TURBINE_VARIABLES:
            assert var.node_id in mapped_node_ids, (
                f"Turbine variable {var.browse_name} (id={var.node_id}) "
                f"has no protobuf field mapping"
            )

    def test_turbine_field_mapping_no_duplicate_node_ids(self) -> None:
        node_ids = list(TURBINE_FIELD_TO_NODEID.values())
        assert len(node_ids) == len(set(node_ids))

    def test_turbine_field_mapping_count(self) -> None:
        assert len(TURBINE_FIELD_TO_NODEID) == 4

    def test_turbine_field_keys_are_valid_proto_fields(self) -> None:
        """All keys must be real TurbineStateMsg field names."""
        dummy = pb.TurbineStateMsg()
        for field in TURBINE_FIELD_TO_NODEID:
            assert hasattr(dummy, field), f"'{field}' is not a TurbineStateMsg field"


# ─── MQTTOPCBridge fixtures ───────────────────────────────────────────────────


@pytest.fixture
def opc() -> MagicMock:
    return make_mock_opc()


@pytest.fixture
def bridge(opc: MagicMock) -> MQTTOPCBridge:
    return MQTTOPCBridge(opc_server=opc)


# ─── MQTTOPCBridge tests ─────────────────────────────────────────────────────


class TestMQTTOPCBridge:
    @pytest.mark.asyncio
    async def test_boiler_topic_calls_update_variable(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, make_boiler_payload())
        assert opc.update_variable.await_count == 5  # 5 boiler fields

    @pytest.mark.asyncio
    async def test_boiler_pressure_mapped_to_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, make_boiler_payload(14_200_000.0))
        # Collect all (node_id, value) calls
        calls = {args[0]: args[1] for args, _ in opc.update_variable.call_args_list}
        assert NODEID_PRESSURE in calls
        assert calls[NODEID_PRESSURE] == pytest.approx(14_200_000.0)

    @pytest.mark.asyncio
    async def test_turbine_topic_calls_update_variable(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        assert opc.update_variable.await_count == 4  # 4 turbine fields

    @pytest.mark.asyncio
    async def test_turbine_electrical_power_mapped_to_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(
            TOPIC_TURBINE, make_turbine_payload(electrical_power_w=210_000_000.0)
        )
        calls = {args[0]: args[1] for args, _ in opc.update_variable.call_args_list}
        assert NODEID_ELECTRICAL_POWER in calls
        assert calls[NODEID_ELECTRICAL_POWER] == pytest.approx(210_000_000.0)

    @pytest.mark.asyncio
    async def test_turbine_shaft_power_mapped_to_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        calls = {args[0]: args[1] for args, _ in opc.update_variable.call_args_list}
        assert NODEID_SHAFT_POWER in calls

    @pytest.mark.asyncio
    async def test_turbine_steam_flow_mapped_to_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        calls = {args[0]: args[1] for args, _ in opc.update_variable.call_args_list}
        assert NODEID_STEAM_FLOW in calls

    @pytest.mark.asyncio
    async def test_turbine_exhaust_pressure_mapped_to_correct_node_id(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        calls = {args[0]: args[1] for args, _ in opc.update_variable.call_args_list}
        assert NODEID_EXHAUST_PRESSURE in calls

    @pytest.mark.asyncio
    async def test_heartbeat_topic_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_HEARTBEAT, b"1741000000000")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_topic_is_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/unknown/variable", b"\x00\x01\x02")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_protobuf_boiler_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, b"not-protobuf-\xff\xfe")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_protobuf_turbine_skipped(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, b"not-protobuf-\xff\xfe")
        opc.update_variable.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_received_counter_increments_on_boiler(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, make_boiler_payload())
        assert bridge.stats["received"] == 1

    @pytest.mark.asyncio
    async def test_received_counter_increments_on_turbine(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        assert bridge.stats["received"] == 1

    @pytest.mark.asyncio
    async def test_mapped_counter_increments_on_boiler_success(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, make_boiler_payload())
        assert bridge.stats["mapped"] == 1

    @pytest.mark.asyncio
    async def test_mapped_counter_increments_on_turbine_success(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        assert bridge.stats["mapped"] == 1

    @pytest.mark.asyncio
    async def test_skipped_counter_increments_on_heartbeat(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_HEARTBEAT, b"12345")
        assert bridge.stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_skipped_counter_increments_on_unknown_topic(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message("sensors/unknown/xyz", b"data")
        assert bridge.stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_skipped_counter_increments_on_bad_protobuf(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, b"\xff\xfe\xfd")
        assert bridge.stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_both_topics_mapped_independently(
        self, bridge: MQTTOPCBridge, opc: MagicMock
    ) -> None:
        await bridge._handle_message(TOPIC_BOILER, make_boiler_payload())
        await bridge._handle_message(TOPIC_TURBINE, make_turbine_payload())
        assert bridge.stats["mapped"] == 2
        # 5 boiler fields + 4 turbine fields = 9 total node updates
        assert opc.update_variable.await_count == 9
