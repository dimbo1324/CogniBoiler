"""
OPC UA address space definition for CogniBoiler.

Builds the node tree under Objects/CogniBoiler/ following
IEC 62541 conventions:
  - All process variables are VariableNodes with engineering units
  - Folders group related variables (Boiler, Turbine)
  - NodeIds use namespace index 2 (ns=2), integer identifiers

Node ID allocation:
    2000  — CogniBoiler (root folder)
    2001  — Boiler (folder)
    2002  — Turbine (folder)
    2100  — Boiler/Pressure
    2101  — Boiler/WaterLevel
    2102  — Boiler/WaterTemp
    2103  — Boiler/FlueGasTemp
    2104  — Boiler/InternalEnergy
    2200  — Turbine/ElectricalPower
    2201  — Turbine/ShaftPower
    2202  — Turbine/SteamFlow
    2203  — Turbine/ExhaustPressure
"""

from __future__ import annotations

from dataclasses import dataclass

# ─── Node ID constants ────────────────────────────────────────────────────────

NS_IDX: int = 2  # Our custom namespace index

# Folder nodes
NODEID_ROOT: int = 2000
NODEID_BOILER_FOLDER: int = 2001
NODEID_TURBINE_FOLDER: int = 2002

# Boiler variable nodes
NODEID_PRESSURE: int = 2100
NODEID_WATER_LEVEL: int = 2101
NODEID_WATER_TEMP: int = 2102
NODEID_FLUE_GAS_TEMP: int = 2103
NODEID_INTERNAL_ENERGY: int = 2104

# Turbine variable nodes
NODEID_ELECTRICAL_POWER: int = 2200
NODEID_SHAFT_POWER: int = 2201
NODEID_STEAM_FLOW: int = 2202
NODEID_EXHAUST_PRESSURE: int = 2203

NAMESPACE_URI: str = "urn:cogniboiler:simulation"


# ─── Variable descriptor ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariableDescriptor:
    """
    Describes a single OPC UA VariableNode.

    Attributes:
        node_id:       Integer node ID (used with NS_IDX).
        browse_name:   Human-readable name shown in OPC UA clients.
        display_name:  Longer display name.
        unit:          Engineering unit string (shown in EURange description).
        description:   What this variable represents.
        initial_value: Value before first physics update arrives.
    """

    node_id: int
    browse_name: str
    display_name: str
    unit: str
    description: str
    initial_value: float = 0.0


# ─── Variable catalogue ───────────────────────────────────────────────────────

BOILER_VARIABLES: list[VariableDescriptor] = [
    VariableDescriptor(
        node_id=NODEID_PRESSURE,
        browse_name="Pressure",
        display_name="Drum Pressure",
        unit="Pa",
        description="Steam drum pressure [Pa]",
        initial_value=140.0e5,
    ),
    VariableDescriptor(
        node_id=NODEID_WATER_LEVEL,
        browse_name="WaterLevel",
        display_name="Water Level",
        unit="m",
        description="Water level in drum [m]",
        initial_value=4.8,
    ),
    VariableDescriptor(
        node_id=NODEID_WATER_TEMP,
        browse_name="WaterTemp",
        display_name="Water Temperature",
        unit="K",
        description="Bulk water temperature [K]",
        initial_value=611.0,
    ),
    VariableDescriptor(
        node_id=NODEID_FLUE_GAS_TEMP,
        browse_name="FlueGasTemp",
        display_name="Flue Gas Temperature",
        unit="K",
        description="Furnace flue gas temperature [K]",
        initial_value=1200.0,
    ),
    VariableDescriptor(
        node_id=NODEID_INTERNAL_ENERGY,
        browse_name="InternalEnergy",
        display_name="Internal Energy",
        unit="J",
        description="Total drum internal energy [J]",
        initial_value=0.0,
    ),
]

TURBINE_VARIABLES: list[VariableDescriptor] = [
    VariableDescriptor(
        node_id=NODEID_ELECTRICAL_POWER,
        browse_name="ElectricalPower",
        display_name="Electrical Power",
        unit="W",
        description="Net electrical output [W]",
        initial_value=0.0,
    ),
    VariableDescriptor(
        node_id=NODEID_SHAFT_POWER,
        browse_name="ShaftPower",
        display_name="Shaft Power",
        unit="W",
        description="Mechanical shaft power [W]",
        initial_value=0.0,
    ),
    VariableDescriptor(
        node_id=NODEID_STEAM_FLOW,
        browse_name="SteamFlow",
        display_name="Steam Flow",
        unit="kg/s",
        description="Steam mass flow through turbine [kg/s]",
        initial_value=0.0,
    ),
    VariableDescriptor(
        node_id=NODEID_EXHAUST_PRESSURE,
        browse_name="ExhaustPressure",
        display_name="Exhaust Pressure",
        unit="Pa",
        description="Condenser back-pressure [Pa]",
        initial_value=7000.0,
    ),
]

ALL_VARIABLES: list[VariableDescriptor] = BOILER_VARIABLES + TURBINE_VARIABLES

# ─── MQTT topic → NodeId mapping ──────────────────────────────────────────────
# Used by the MQTT subscriber to know which OPC UA node to update.

MQTT_TOPIC_TO_NODEID: dict[str, int] = {
    "sensors/boiler/pressure_pa": NODEID_PRESSURE,
    "sensors/boiler/water_level_m": NODEID_WATER_LEVEL,
    "sensors/boiler/water_temp_k": NODEID_WATER_TEMP,
    "sensors/boiler/flue_gas_temp_k": NODEID_FLUE_GAS_TEMP,
    "sensors/boiler/internal_energy_j": NODEID_INTERNAL_ENERGY,
    "sensors/turbine/electrical_power_w": NODEID_ELECTRICAL_POWER,
    "sensors/turbine/shaft_power_w": NODEID_SHAFT_POWER,
    "sensors/turbine/steam_flow_kg_s": NODEID_STEAM_FLOW,
    "sensors/turbine/exhaust_pressure_pa": NODEID_EXHAUST_PRESSURE,
}
