"""
Protobuf mapping of plant state — the one place physics types meet the contract.

Telemetry carries measured values: an instrument fault changes what is published, the
true state stays inside the simulator. Faults are published with their labels.
"""

from __future__ import annotations

import time

import cogniboiler_pb2 as pb

from physics_engine.constants import FUEL_HEATING_VALUE
from physics_engine.faults import ActiveFault, FaultKind
from physics_engine.models import BoilerState, ControlInputs
from physics_engine.plant import PlantSnapshot
from physics_engine.runtime import RunState, SimulationStatus
from physics_engine.scenarios import ScenarioDefinition
from physics_engine.sensors import SensorId, SensorReading, worst_quality
from physics_engine.turbine import TurbineState

BOILER_SENSORS: tuple[SensorId, ...] = (
    SensorId.DRUM_PRESSURE,
    SensorId.DRUM_LEVEL,
    SensorId.DRUM_WATER_TEMP,
    SensorId.FURNACE_GAS_TEMP,
    SensorId.FEEDWATER_FLOW,
    SensorId.FUEL_FLOW,
)


def now_ms() -> int:
    """Current UTC epoch milliseconds."""
    return int(time.time() * 1000)


def fault_kind_to_proto(kind: FaultKind) -> int:
    return int(pb.FaultKind.Value(f"FAULT_{kind.value.upper()}"))


def fault_kind_from_proto(value: int) -> FaultKind:
    """Map a protobuf FaultKind to the plant's kind, raising ValueError if unknown."""
    name = pb.FaultKind.Name(value)
    if not name.startswith("FAULT_") or name == "FAULT_KIND_UNSPECIFIED":
        raise ValueError(f"unsupported fault kind {name}")
    return FaultKind(name.removeprefix("FAULT_").lower())


def boiler_state_to_proto(
    state: BoilerState,
    quality: int = pb.SensorQuality.GOOD,
) -> pb.BoilerStateMsg:
    """
    Convert a BoilerState dataclass to a BoilerStateMsg protobuf message.

    Args:
        state:   Boiler state (true or as measured).
        quality: Worst instrument quality of the boiler readings.

    Returns:
        Populated BoilerStateMsg ready for serialization.
    """
    return pb.BoilerStateMsg(
        pressure_pa=state.pressure,
        water_level_m=state.water_level,
        water_temp_k=state.water_temp,
        flue_gas_temp_k=state.flue_gas_temp,
        internal_energy_j=state.internal_energy,
        timestamp_ms=now_ms(),
        quality=quality,
    )


def turbine_state_to_proto(state: TurbineState) -> pb.TurbineStateMsg:
    """
    Convert a TurbineState dataclass to a TurbineStateMsg protobuf message.

    Args:
        state: Current turbine state.

    Returns:
        Populated TurbineStateMsg ready for serialization.
    """
    return pb.TurbineStateMsg(
        electrical_power_w=state.electrical_power,
        shaft_power_w=state.shaft_power,
        enthalpy_in_j_kg=state.enthalpy_in,
        enthalpy_out_j_kg=state.enthalpy_out_actual,
        exhaust_pressure_pa=state.exhaust_pressure,
        steam_flow_kg_s=state.steam_flow,
        timestamp_ms=now_ms(),
        steam_temp_in_k=state.steam_temp_in,
        exhaust_temp_k=state.exhaust_temp,
    )


def boiler_to_proto(snapshot: PlantSnapshot) -> pb.BoilerStateMsg:
    """Measured boiler state with flows and heat duties."""
    readings = [snapshot.reading(sensor) for sensor in BOILER_SENSORS]
    message = boiler_state_to_proto(
        snapshot.measured_boiler(), quality=int(worst_quality(readings))
    )
    message.fuel_flow_kg_s = snapshot.measured(SensorId.FUEL_FLOW)
    message.feedwater_flow_kg_s = snapshot.measured(SensorId.FEEDWATER_FLOW)
    message.drum_steam_flow_kg_s = snapshot.flows.drum_steam
    message.spray_flow_kg_s = snapshot.flows.spray
    message.relief_flow_kg_s = snapshot.flows.relief
    message.superheater_outlet_temp_k = snapshot.heat.superheater_outlet_temp
    message.economizer_outlet_temp_k = snapshot.heat.economizer_outlet_temp
    message.stack_temp_k = snapshot.heat.stack_temp
    message.heat_release_w = snapshot.heat.heat_release
    message.boiler_efficiency = snapshot.heat.boiler_efficiency
    message.feedwater_temp_k = snapshot.heat.feedwater_temp
    return message


def turbine_to_proto(snapshot: PlantSnapshot) -> pb.TurbineStateMsg:
    """Turbine state with its instrumented values as measured."""
    return turbine_state_to_proto(snapshot.measured_turbine())


def actuators_to_proto(controls: ControlInputs) -> pb.ActuatorStateMsg:
    """Valve commands and actual positions."""
    return pb.ActuatorStateMsg(
        fuel_valve_command=controls.fuel_valve_command,
        fuel_valve_position=controls.fuel_valve.position,
        feedwater_valve_command=controls.feedwater_valve_command,
        feedwater_valve_position=controls.feedwater_valve.position,
        steam_valve_command=controls.steam_valve_command,
        steam_valve_position=controls.steam_valve.position,
        spray_valve_command=controls.spray_valve_command,
        spray_valve_position=controls.spray_valve.position,
    )


def emissions_to_proto(snapshot: PlantSnapshot) -> pb.EmissionsMsg:
    emissions = snapshot.emissions
    power_mw = snapshot.turbine.electrical_power_mw
    intensity = emissions.co2_rate * 3600.0 / power_mw if power_mw > 1.0 else 0.0
    return pb.EmissionsMsg(
        co2_kg_s=emissions.co2_rate,
        nox_kg_s=emissions.nox_rate,
        co_kg_s=emissions.co_rate,
        nox_ppmv=emissions.nox_ppmv,
        co2_intensity_kg_per_mwh=intensity,
    )


# Below this output the heat rates and intensities are meaningless and reported as zero.
MIN_GENERATING_POWER_W: float = 1.0e6


def performance_to_proto(snapshot: PlantSnapshot) -> pb.PerformanceMsg:
    """Efficiency and heat rates from the true heat balance of the step."""
    fuel_heat_input = snapshot.flows.fuel * FUEL_HEATING_VALUE
    heat_to_cycle = snapshot.heat.boiler_efficiency * fuel_heat_input
    power = snapshot.turbine.electrical_power
    if power < MIN_GENERATING_POWER_W or fuel_heat_input <= 0.0:
        return pb.PerformanceMsg(
            fuel_heat_input_w=fuel_heat_input,
            heat_to_cycle_w=heat_to_cycle,
            boiler_efficiency=snapshot.heat.boiler_efficiency,
            electrical_power_w=power,
        )
    return pb.PerformanceMsg(
        fuel_heat_input_w=fuel_heat_input,
        heat_to_cycle_w=heat_to_cycle,
        boiler_efficiency=snapshot.heat.boiler_efficiency,
        electrical_power_w=power,
        net_efficiency=power / fuel_heat_input,
        turbine_heat_rate_j_per_j=heat_to_cycle / power,
        plant_heat_rate_j_per_j=fuel_heat_input / power,
        co2_intensity_kg_per_j=snapshot.emissions.co2_rate / power,
    )


def condenser_to_proto(snapshot: PlantSnapshot) -> pb.CondenserMsg:
    condenser = snapshot.condenser
    return pb.CondenserMsg(
        backpressure_pa=condenser.backpressure_pa,
        condensate_temp_k=condenser.condensate_temp,
        cooling_water_temp_in_k=snapshot.cooling_water_temp_k,
        cooling_water_temp_out_k=condenser.cooling_water_temp_out,
        heat_rejected_w=condenser.heat_rejected_w,
        loading=condenser.condenser_loading,
    )


def health_to_proto(snapshot: PlantSnapshot) -> pb.EquipmentHealthMsg:
    health = snapshot.health
    return pb.EquipmentHealthMsg(
        turbine_hours=health.turbine_hours,
        turbine_starts=health.turbine_starts,
        turbine_damage=health.turbine_damage,
        boiler_tube_hours=health.boiler_tube_hours,
        boiler_tube_damage=health.boiler_tube_damage,
        pump_hours=health.pump_hours,
        overall_health_pct=health.overall_health_pct,
        maintenance_alarm=health.maintenance_alarm,
        maintenance_critical=health.maintenance_critical,
    )


def fault_to_proto(fault: ActiveFault, simulation_time_s: float) -> pb.FaultMsg:
    return pb.FaultMsg(
        fault_id=fault.fault_id,
        kind=fault_kind_to_proto(fault.spec.kind),
        target=fault.spec.target,
        severity=fault.spec.severity,
        ramp_s=fault.spec.ramp_s,
        started_at_s=fault.started_at_s,
        intensity=fault.intensity(simulation_time_s),
        label=fault.label,
    )


def sensor_to_proto(reading: SensorReading) -> pb.SensorStatusMsg:
    return pb.SensorStatusMsg(
        sensor_id=reading.sensor_id.value,
        quality=int(reading.quality),
        measured_value=reading.measured_value,
    )


def simulation_status_to_proto(status: SimulationStatus) -> pb.SimulationStatusMsg:
    return pb.SimulationStatusMsg(
        run_state=(
            pb.SimulationRunState.SIMULATION_PAUSED
            if status.run_state is RunState.PAUSED
            else pb.SimulationRunState.SIMULATION_RUNNING
        ),
        speed_factor=status.speed_factor,
        simulation_time_s=status.simulation_time_s,
        step_count=status.step_count,
        scenario=status.scenario.value,
        run_id=status.run_id,
        step_s=status.step_s,
    )


def scenario_to_proto(definition: ScenarioDefinition) -> pb.ScenarioMsg:
    return pb.ScenarioMsg(
        name=definition.name.value,
        title=definition.title,
        description=definition.description,
    )


def system_state_to_proto(
    snapshot: PlantSnapshot, status: SimulationStatus
) -> pb.SystemStateMsg:
    """Full plant state as the PhysicsService returns it."""
    return pb.SystemStateMsg(
        boiler=boiler_to_proto(snapshot),
        turbine=turbine_to_proto(snapshot),
        actuators=actuators_to_proto(snapshot.controls),
        simulation_time_s=snapshot.simulation_time_s,
        emissions=emissions_to_proto(snapshot),
        condenser=condenser_to_proto(snapshot),
        health=health_to_proto(snapshot),
        active_faults=[
            fault_to_proto(fault, snapshot.simulation_time_s)
            for fault in snapshot.faults
        ],
        sensors=[sensor_to_proto(reading) for reading in snapshot.sensors],
        simulation=simulation_status_to_proto(status),
        performance=performance_to_proto(snapshot),
    )


def plant_status_to_proto(
    snapshot: PlantSnapshot, status: SimulationStatus
) -> pb.PlantStatusMsg:
    """The plant telemetry that is neither boiler nor turbine state."""
    return pb.PlantStatusMsg(
        emissions=emissions_to_proto(snapshot),
        condenser=condenser_to_proto(snapshot),
        health=health_to_proto(snapshot),
        active_faults=[
            fault_to_proto(fault, snapshot.simulation_time_s)
            for fault in snapshot.faults
        ],
        sensors=[sensor_to_proto(reading) for reading in snapshot.sensors],
        simulation=simulation_status_to_proto(status),
        actuators=actuators_to_proto(snapshot.controls),
        timestamp_ms=now_ms(),
        performance=performance_to_proto(snapshot),
    )
