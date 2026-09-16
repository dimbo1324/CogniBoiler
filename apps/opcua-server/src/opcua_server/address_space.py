"""
OPC UA address space definition for CogniBoiler.

Objects/CogniBoiler/ in namespace urn:cogniboiler:simulation (ns=2), integer NodeIds:

    2000        CogniBoiler (root folder)
    2001–2010   folders: Boiler, Turbine, Valves, Emissions, Performance, Condenser,
                Health, Simulation, PLC, Alarms
    2100–2199   Boiler      measured drum values, flows, temperatures, heat, efficiency
    2200–2299   Turbine     measured output, steam and exhaust
    2300–2399   Valves      command and actual position of fuel, feedwater, steam, spray
    2400–2449   Emissions
    2450–2499   Condenser
    2500–2549   Performance KPIs computed by the physics engine
    2550–2599   Health      equipment wear and maintenance flags
    2600–2699   Simulation  scenario, run, time, speed, active faults
    2700–2799   PLC         mode, E-Stop, trip cause, reset permission, targets
    2800–2899   Alarms      counts and the open alarms
    2900–2999   methods

Node ids of the first release (2100–2104, 2200–2203) are unchanged. Every variable is
read-only for clients; values change only from the plant, the PLC and the alarm service.
Numeric values are in SI units (the unit is the EngineeringUnits property); instrument
quality is carried as the OPC UA status code of the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

NS_IDX: int = 2  # Our custom namespace index
NAMESPACE_URI: str = "urn:cogniboiler:simulation"

# ─── Folders ──────────────────────────────────────────────────────────────────

NODEID_ROOT: int = 2000
NODEID_BOILER_FOLDER: int = 2001
NODEID_TURBINE_FOLDER: int = 2002
NODEID_VALVES_FOLDER: int = 2003
NODEID_EMISSIONS_FOLDER: int = 2004
NODEID_PERFORMANCE_FOLDER: int = 2005
NODEID_CONDENSER_FOLDER: int = 2006
NODEID_HEALTH_FOLDER: int = 2007
NODEID_SIMULATION_FOLDER: int = 2008
NODEID_PLC_FOLDER: int = 2009
NODEID_ALARMS_FOLDER: int = 2010


class Folder(StrEnum):
    BOILER = "Boiler"
    TURBINE = "Turbine"
    VALVES = "Valves"
    EMISSIONS = "Emissions"
    PERFORMANCE = "Performance"
    CONDENSER = "Condenser"
    HEALTH = "Health"
    SIMULATION = "Simulation"
    PLC = "PLC"
    ALARMS = "Alarms"


FOLDER_NODE_IDS: dict[Folder, int] = {
    Folder.BOILER: NODEID_BOILER_FOLDER,
    Folder.TURBINE: NODEID_TURBINE_FOLDER,
    Folder.VALVES: NODEID_VALVES_FOLDER,
    Folder.EMISSIONS: NODEID_EMISSIONS_FOLDER,
    Folder.PERFORMANCE: NODEID_PERFORMANCE_FOLDER,
    Folder.CONDENSER: NODEID_CONDENSER_FOLDER,
    Folder.HEALTH: NODEID_HEALTH_FOLDER,
    Folder.SIMULATION: NODEID_SIMULATION_FOLDER,
    Folder.PLC: NODEID_PLC_FOLDER,
    Folder.ALARMS: NODEID_ALARMS_FOLDER,
}

# ─── Variables of the first release ──────────────────────────────────────────

NODEID_PRESSURE: int = 2100
NODEID_WATER_LEVEL: int = 2101
NODEID_WATER_TEMP: int = 2102
NODEID_FLUE_GAS_TEMP: int = 2103
NODEID_INTERNAL_ENERGY: int = 2104

NODEID_ELECTRICAL_POWER: int = 2200
NODEID_SHAFT_POWER: int = 2201
NODEID_STEAM_FLOW: int = 2202
NODEID_EXHAUST_PRESSURE: int = 2203

# ─── Methods ──────────────────────────────────────────────────────────────────

NODEID_METHOD_SET_LOAD_DEMAND: int = 2900
NODEID_METHOD_SET_CONTROL_MODE: int = 2901
NODEID_METHOD_RESET_EMERGENCY_STOP: int = 2902
NODEID_METHOD_APPLY_VALVE_COMMAND: int = 2903
NODEID_METHOD_ACKNOWLEDGE_ALARM: int = 2910
NODEID_METHOD_ACKNOWLEDGE_ALL_ALARMS: int = 2911


class ValueKind(StrEnum):
    DOUBLE = "Double"
    BOOLEAN = "Boolean"
    STRING = "String"
    INT64 = "Int64"
    STRING_ARRAY = "StringArray"


InitialValue = float | bool | str | int | list[str]


@dataclass(frozen=True)
class VariableDescriptor:
    """
    Describes a single OPC UA VariableNode.

    Attributes:
        node_id:       Integer node ID (used with NS_IDX).
        browse_name:   Name shown in OPC UA clients; unique in the whole tree.
        display_name:  Longer display name.
        unit:          Engineering unit; "1" for fractions, "-" for text and flags.
        description:   What this variable represents.
        initial_value: Value before the first update arrives.
        kind:          OPC UA value type.
        folder:        Folder the node lives in.
    """

    node_id: int
    browse_name: str
    display_name: str
    unit: str
    description: str
    initial_value: InitialValue = 0.0
    kind: ValueKind = ValueKind.DOUBLE
    folder: Folder = Folder.BOILER


def _d(
    node_id: int,
    browse_name: str,
    display_name: str,
    unit: str,
    description: str,
    folder: Folder,
    initial_value: InitialValue = 0.0,
    kind: ValueKind = ValueKind.DOUBLE,
) -> VariableDescriptor:
    return VariableDescriptor(
        node_id=node_id,
        browse_name=browse_name,
        display_name=display_name,
        unit=unit,
        description=description,
        initial_value=initial_value,
        kind=kind,
        folder=folder,
    )


B, T, V, E, P, C, H, S, L, A = (
    Folder.BOILER,
    Folder.TURBINE,
    Folder.VALVES,
    Folder.EMISSIONS,
    Folder.PERFORMANCE,
    Folder.CONDENSER,
    Folder.HEALTH,
    Folder.SIMULATION,
    Folder.PLC,
    Folder.ALARMS,
)
BOOL, TEXT, INT, TEXTS = (
    ValueKind.BOOLEAN,
    ValueKind.STRING,
    ValueKind.INT64,
    ValueKind.STRING_ARRAY,
)

# ─── Catalogue ────────────────────────────────────────────────────────────────

BOILER_VARIABLES: list[VariableDescriptor] = [
    _d(NODEID_PRESSURE, "Pressure", "Drum Pressure", "Pa", "Steam drum pressure [Pa]", B, 140.0e5),
    _d(NODEID_WATER_LEVEL, "WaterLevel", "Water Level", "m", "Water level in drum [m]", B, 4.8),
    _d(NODEID_WATER_TEMP, "WaterTemp", "Water Temperature", "K", "Bulk water temperature [K]", B, 611.0),
    _d(NODEID_FLUE_GAS_TEMP, "FlueGasTemp", "Flue Gas Temperature", "K", "Furnace flue gas temperature [K]", B, 1200.0),
    _d(NODEID_INTERNAL_ENERGY, "InternalEnergy", "Internal Energy", "J", "Total drum internal energy [J]", B),
    _d(2105, "FuelFlow", "Fuel Flow", "kg/s", "Measured fuel mass flow [kg/s]", B),
    _d(2106, "FeedwaterFlow", "Feedwater Flow", "kg/s", "Measured feedwater mass flow [kg/s]", B),
    _d(2107, "DrumSteamFlow", "Drum Steam Flow", "kg/s", "Steam raised in the drum [kg/s]", B),
    _d(2108, "SprayFlow", "Spray Flow", "kg/s", "Attemperator spray water flow [kg/s]", B),
    _d(2109, "SuperheaterOutletTemp", "Superheater Outlet Temperature", "K", "Steam temperature before the spray [K]", B),
    _d(2110, "EconomizerOutletTemp", "Economizer Outlet Temperature", "K", "Feedwater temperature after the economizer [K]", B),
    _d(2111, "StackTemp", "Stack Temperature", "K", "Flue gas temperature at the stack [K]", B),
    _d(2112, "HeatRelease", "Heat Release", "W", "Heat released in the furnace [W]", B),
    _d(2113, "BoilerEfficiency", "Boiler Efficiency", "1", "Heat to water and steam over fuel heat input [1]", B),
    _d(2114, "FeedwaterTemp", "Feedwater Temperature", "K", "Feedwater inlet temperature [K]", B),
    _d(2115, "ReliefFlow", "Relief Flow", "kg/s", "Steam discharged by the drum safety valves [kg/s]", B),
]  # fmt: skip

TURBINE_VARIABLES: list[VariableDescriptor] = [
    _d(NODEID_ELECTRICAL_POWER, "ElectricalPower", "Electrical Power", "W", "Net electrical output [W]", T),
    _d(NODEID_SHAFT_POWER, "ShaftPower", "Shaft Power", "W", "Mechanical shaft power [W]", T),
    _d(NODEID_STEAM_FLOW, "SteamFlow", "Steam Flow", "kg/s", "Steam mass flow through turbine [kg/s]", T),
    _d(NODEID_EXHAUST_PRESSURE, "ExhaustPressure", "Exhaust Pressure", "Pa", "Condenser back-pressure [Pa]", T, 7000.0),
    _d(2204, "SteamTempIn", "Inlet Steam Temperature", "K", "Turbine inlet steam temperature [K]", T),
    _d(2205, "ExhaustTemp", "Exhaust Temperature", "K", "Turbine exhaust temperature [K]", T),
    _d(2206, "EnthalpyIn", "Inlet Enthalpy", "J/kg", "Inlet specific enthalpy [J/kg]", T),
    _d(2207, "EnthalpyOut", "Outlet Enthalpy", "J/kg", "Outlet specific enthalpy [J/kg]", T),
]  # fmt: skip

VALVE_VARIABLES: list[VariableDescriptor] = [
    _d(2300, "FuelValveCommand", "Fuel Valve Command", "1", "Commanded fuel valve opening [0..1]", V),
    _d(2301, "FuelValvePosition", "Fuel Valve Position", "1", "Actual fuel valve opening [0..1]", V),
    _d(2302, "FeedwaterValveCommand", "Feedwater Valve Command", "1", "Commanded feedwater valve opening [0..1]", V),
    _d(2303, "FeedwaterValvePosition", "Feedwater Valve Position", "1", "Actual feedwater valve opening [0..1]", V),
    _d(2304, "SteamValveCommand", "Steam Valve Command", "1", "Commanded turbine admission valve opening [0..1]", V),
    _d(2305, "SteamValvePosition", "Steam Valve Position", "1", "Actual turbine admission valve opening [0..1]", V),
    _d(2306, "SprayValveCommand", "Spray Valve Command", "1", "Commanded attemperator spray valve opening [0..1]", V),
    _d(2307, "SprayValvePosition", "Spray Valve Position", "1", "Actual attemperator spray valve opening [0..1]", V),
]  # fmt: skip

EMISSIONS_VARIABLES: list[VariableDescriptor] = [
    _d(2400, "CO2Flow", "CO2 Mass Flow", "kg/s", "CO2 emitted at the stack [kg/s]", E),
    _d(2401, "NOxFlow", "NOx Mass Flow", "kg/s", "NOx (as NO2) emitted at the stack [kg/s]", E),
    _d(2402, "COFlow", "CO Mass Flow", "kg/s", "CO emitted at the stack [kg/s]", E),
    _d(2403, "NOxConcentration", "NOx Concentration", "ppmv", "Dry stack NOx at 3 % O2 [ppmv]", E),
    _d(2404, "CO2IntensityPerMWh", "CO2 Intensity", "kg/MWh", "CO2 per MWh of electricity [kg/MWh]", E),
]  # fmt: skip

CONDENSER_VARIABLES: list[VariableDescriptor] = [
    _d(2450, "CondenserBackpressure", "Condenser Back-pressure", "Pa", "Condenser pressure [Pa]", C),
    _d(2451, "CondensateTemp", "Condensate Temperature", "K", "Condensate temperature [K]", C),
    _d(2452, "CoolingWaterInletTemp", "Cooling Water Inlet Temperature", "K", "Cooling water inlet temperature [K]", C),
    _d(2453, "CoolingWaterOutletTemp", "Cooling Water Outlet Temperature", "K", "Cooling water outlet temperature [K]", C),
    _d(2454, "HeatRejected", "Heat Rejected", "W", "Heat rejected to cooling water [W]", C),
    _d(2455, "CondenserLoading", "Condenser Loading", "1", "Heat duty over design duty [1]", C),
]  # fmt: skip

PERFORMANCE_VARIABLES: list[VariableDescriptor] = [
    _d(2500, "FuelHeatInput", "Fuel Heat Input", "W", "Fuel mass flow times lower heating value [W]", P),
    _d(2501, "HeatToCycle", "Heat to Cycle", "W", "Heat absorbed by water and steam [W]", P),
    _d(2502, "NetEfficiency", "Net Efficiency", "1", "Electrical output over fuel heat input [1]", P),
    _d(2503, "TurbineHeatRate", "Turbine Heat Rate", "J/J", "Heat to cycle per unit of electricity [J/J]", P),
    _d(2504, "PlantHeatRate", "Plant Heat Rate", "J/J", "Fuel heat per unit of electricity [J/J]", P),
    _d(2505, "CO2Intensity", "CO2 Intensity per Joule", "kg/J", "CO2 per unit of electricity [kg/J]", P),
    _d(2506, "TrueElectricalPower", "True Electrical Power", "W", "Electrical output of the model, not the reading [W]", P),
]  # fmt: skip

HEALTH_VARIABLES: list[VariableDescriptor] = [
    _d(2550, "OverallHealth", "Overall Health", "%", "Equipment health index [%]", H, 100.0),
    _d(2551, "TurbineHours", "Turbine Hours", "h", "Turbine operating hours [h]", H),
    _d(2552, "TurbineStarts", "Turbine Starts", "1", "Turbine starts [1]", H),
    _d(2553, "TurbineDamage", "Turbine Damage", "1", "Miner damage fraction of design life [1]", H),
    _d(2554, "BoilerTubeHours", "Boiler Tube Hours", "h", "Boiler tube hours at temperature [h]", H),
    _d(2555, "BoilerTubeDamage", "Boiler Tube Damage", "1", "Tube damage fraction of design life [1]", H),
    _d(2556, "PumpHours", "Pump Hours", "h", "Feedwater pump operating hours [h]", H),
    _d(2557, "MaintenanceAlarm", "Maintenance Alarm", "-", "Maintenance is due", H, False, BOOL),
    _d(2558, "MaintenanceCritical", "Maintenance Critical", "-", "Maintenance is overdue", H, False, BOOL),
]  # fmt: skip

SIMULATION_VARIABLES: list[VariableDescriptor] = [
    _d(2600, "Scenario", "Scenario", "-", "Scenario the plant was started from", S, "", TEXT),
    _d(2601, "RunId", "Run Id", "-", "Changes whenever a scenario is loaded", S, 0, INT),
    _d(2602, "SimulationTime", "Simulation Time", "s", "Simulated time since the scenario was loaded [s]", S),
    _d(2603, "SpeedFactor", "Speed Factor", "1", "Simulated seconds per wall-clock second [1]", S, 1.0),
    _d(2604, "Paused", "Paused", "-", "The simulation is paused", S, False, BOOL),
    _d(2605, "ActiveFaults", "Active Faults", "-", "Labels of the injected faults", S, [], TEXTS),
    _d(2606, "InstrumentsNotGood", "Instruments Not Good", "1", "Instruments reporting uncertain or bad quality [1]", S, 0, INT),
]  # fmt: skip

PLC_VARIABLES: list[VariableDescriptor] = [
    _d(2700, "Mode", "Control Mode", "-", "auto | manual | estop", L, "", TEXT),
    _d(2701, "EmergencyStopActive", "Emergency Stop Active", "-", "The E-Stop latch is set", L, False, BOOL),
    _d(2702, "TripCause", "Trip Cause", "-", "Parameter, value and limit of the latched trip", L, "", TEXT),
    _d(2703, "ResetPermitted", "Reset Permitted", "-", "An E-Stop reset would be accepted now", L, False, BOOL),
    _d(2704, "ResetBlockers", "Reset Blockers", "-", "Why a reset would be refused now", L, [], TEXTS),
    _d(2705, "LoadDemand", "Load Demand", "W", "Electrical load target set by the operator [W]", L),
    _d(2706, "LoadSetpoint", "Load Setpoint", "W", "Ramped working load setpoint [W]", L),
    _d(2707, "PressureSetpoint", "Pressure Setpoint", "Pa", "Drum pressure target [Pa]", L),
    _d(2708, "LevelSetpoint", "Level Setpoint", "m", "Drum level target [m]", L),
    _d(2709, "SteamTempSetpoint", "Steam Temperature Setpoint", "K", "Steam temperature target [K]", L),
    _d(2710, "WarningCount", "Warning Count", "1", "Warnings raised by the interlocks [1]", L, 0, INT),
    _d(2711, "TripCount", "Trip Count", "1", "Trips latched by the interlocks [1]", L, 0, INT),
    _d(2712, "PlcCommunication", "PLC Communication", "-", "The OPC UA server reaches PLCService", L, False, BOOL),
]  # fmt: skip

ALARM_VARIABLES: list[VariableDescriptor] = [
    _d(2800, "OpenAlarmCount", "Open Alarms", "1", "Alarms not both cleared and acknowledged [1]", A, 0, INT),
    _d(2801, "UnacknowledgedCount", "Unacknowledged Alarms", "1", "Open alarms waiting for acknowledgement [1]", A, 0, INT),
    _d(2802, "CriticalActiveCount", "Critical Active Alarms", "1", "Critical alarms whose condition is present [1]", A, 0, INT),
    _d(2803, "OpenAlarms", "Open Alarm List", "-", "id | severity | state | message of each open alarm", A, [], TEXTS),
    _d(2804, "AlarmServiceCommunication", "Alarm Service Communication", "-", "The OPC UA server reaches AlarmService", A, False, BOOL),
]  # fmt: skip

ALL_VARIABLES: list[VariableDescriptor] = (
    BOILER_VARIABLES
    + TURBINE_VARIABLES
    + VALVE_VARIABLES
    + EMISSIONS_VARIABLES
    + CONDENSER_VARIABLES
    + PERFORMANCE_VARIABLES
    + HEALTH_VARIABLES
    + SIMULATION_VARIABLES
    + PLC_VARIABLES
    + ALARM_VARIABLES
)

# ─── Protobuf field -> NodeId mappings ────────────────────────────────────────
# Keys are protobuf field names; getattr(msg, field_name) yields the value.

BOILER_FIELD_TO_NODEID: dict[str, int] = {
    "pressure_pa": NODEID_PRESSURE,
    "water_level_m": NODEID_WATER_LEVEL,
    "water_temp_k": NODEID_WATER_TEMP,
    "flue_gas_temp_k": NODEID_FLUE_GAS_TEMP,
    "internal_energy_j": NODEID_INTERNAL_ENERGY,
    "fuel_flow_kg_s": 2105,
    "feedwater_flow_kg_s": 2106,
    "drum_steam_flow_kg_s": 2107,
    "spray_flow_kg_s": 2108,
    "superheater_outlet_temp_k": 2109,
    "economizer_outlet_temp_k": 2110,
    "stack_temp_k": 2111,
    "heat_release_w": 2112,
    "boiler_efficiency": 2113,
    "feedwater_temp_k": 2114,
    "relief_flow_kg_s": 2115,
}

TURBINE_FIELD_TO_NODEID: dict[str, int] = {
    "electrical_power_w": NODEID_ELECTRICAL_POWER,
    "shaft_power_w": NODEID_SHAFT_POWER,
    "steam_flow_kg_s": NODEID_STEAM_FLOW,
    "exhaust_pressure_pa": NODEID_EXHAUST_PRESSURE,
    "steam_temp_in_k": 2204,
    "exhaust_temp_k": 2205,
    "enthalpy_in_j_kg": 2206,
    "enthalpy_out_j_kg": 2207,
}

ACTUATOR_FIELD_TO_NODEID: dict[str, int] = {
    "fuel_valve_command": 2300,
    "fuel_valve_position": 2301,
    "feedwater_valve_command": 2302,
    "feedwater_valve_position": 2303,
    "steam_valve_command": 2304,
    "steam_valve_position": 2305,
    "spray_valve_command": 2306,
    "spray_valve_position": 2307,
}

EMISSIONS_FIELD_TO_NODEID: dict[str, int] = {
    "co2_kg_s": 2400,
    "nox_kg_s": 2401,
    "co_kg_s": 2402,
    "nox_ppmv": 2403,
    "co2_intensity_kg_per_mwh": 2404,
}

CONDENSER_FIELD_TO_NODEID: dict[str, int] = {
    "backpressure_pa": 2450,
    "condensate_temp_k": 2451,
    "cooling_water_temp_in_k": 2452,
    "cooling_water_temp_out_k": 2453,
    "heat_rejected_w": 2454,
    "loading": 2455,
}

PERFORMANCE_FIELD_TO_NODEID: dict[str, int] = {
    "fuel_heat_input_w": 2500,
    "heat_to_cycle_w": 2501,
    "net_efficiency": 2502,
    "turbine_heat_rate_j_per_j": 2503,
    "plant_heat_rate_j_per_j": 2504,
    "co2_intensity_kg_per_j": 2505,
    "electrical_power_w": 2506,
}

HEALTH_FIELD_TO_NODEID: dict[str, int] = {
    "overall_health_pct": 2550,
    "turbine_hours": 2551,
    "turbine_starts": 2552,
    "turbine_damage": 2553,
    "boiler_tube_hours": 2554,
    "boiler_tube_damage": 2555,
    "pump_hours": 2556,
    "maintenance_alarm": 2557,
    "maintenance_critical": 2558,
}

# Instruments of the plant (SensorStatusMsg.sensor_id) and the nodes they feed; the
# node's status code follows the instrument's quality.
SENSOR_TO_NODEID: dict[str, int] = {
    "drum_pressure": NODEID_PRESSURE,
    "drum_level": NODEID_WATER_LEVEL,
    "drum_water_temp": NODEID_WATER_TEMP,
    "furnace_gas_temp": NODEID_FLUE_GAS_TEMP,
    "fuel_flow": 2105,
    "feedwater_flow": 2106,
    "steam_temp": 2204,
    "steam_flow": NODEID_STEAM_FLOW,
    "electrical_power": NODEID_ELECTRICAL_POWER,
}

VARIABLES_BY_NODE_ID: dict[int, VariableDescriptor] = {
    variable.node_id: variable for variable in ALL_VARIABLES
}
