"""
The full plant snapshot, as REST returns it and the telemetry WebSocket channel streams
it. Field names and units follow the protobuf contract (SI units); enums become
lowercase strings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QualityName = Literal["good", "uncertain", "bad"]


class SimulationStatusResponse(BaseModel):
    run_state: Literal["running", "paused"]
    speed_factor: float
    simulation_time_s: float
    step_count: int
    scenario: str
    run_id: int = Field(..., description="Changes whenever a scenario is loaded.")
    step_s: float


class BoilerStateResponse(BaseModel):
    pressure_pa: float
    water_level_m: float
    water_temp_k: float
    flue_gas_temp_k: float
    internal_energy_j: float
    timestamp_ms: int
    quality: QualityName = Field(..., description="Worst boiler instrument quality.")
    fuel_flow_kg_s: float
    feedwater_flow_kg_s: float
    drum_steam_flow_kg_s: float
    spray_flow_kg_s: float
    superheater_outlet_temp_k: float
    economizer_outlet_temp_k: float
    stack_temp_k: float
    heat_release_w: float
    boiler_efficiency: float
    feedwater_temp_k: float
    relief_flow_kg_s: float


class TurbineStateResponse(BaseModel):
    electrical_power_w: float
    shaft_power_w: float
    enthalpy_in_j_kg: float
    enthalpy_out_j_kg: float
    exhaust_pressure_pa: float
    steam_flow_kg_s: float
    timestamp_ms: int
    steam_temp_in_k: float
    exhaust_temp_k: float


class ActuatorStateResponse(BaseModel):
    fuel_valve_command: float
    fuel_valve_position: float
    feedwater_valve_command: float
    feedwater_valve_position: float
    steam_valve_command: float
    steam_valve_position: float
    spray_valve_command: float
    spray_valve_position: float


class EmissionsResponse(BaseModel):
    co2_kg_s: float
    nox_kg_s: float
    co_kg_s: float
    nox_ppmv: float
    co2_intensity_kg_per_mwh: float


class CondenserResponse(BaseModel):
    backpressure_pa: float
    condensate_temp_k: float
    cooling_water_temp_in_k: float
    cooling_water_temp_out_k: float
    heat_rejected_w: float
    loading: float


class EquipmentHealthResponse(BaseModel):
    turbine_hours: float
    turbine_starts: float
    turbine_damage: float
    boiler_tube_hours: float
    boiler_tube_damage: float
    pump_hours: float
    overall_health_pct: float
    maintenance_alarm: bool
    maintenance_critical: bool


class PerformanceResponse(BaseModel):
    fuel_heat_input_w: float
    heat_to_cycle_w: float
    net_efficiency: float = Field(..., description="0..1; 0 when not generating.")
    turbine_heat_rate_j_per_j: float
    plant_heat_rate_j_per_j: float
    co2_intensity_kg_per_j: float
    boiler_efficiency: float
    electrical_power_w: float = Field(..., description="True output, not the reading.")


class FaultResponse(BaseModel):
    fault_id: str
    kind: str
    target: str
    severity: float
    ramp_s: float
    started_at_s: float
    intensity: float = Field(..., description="0 at onset, 1 once fully developed.")
    label: str


class SensorStatusResponse(BaseModel):
    sensor_id: str
    quality: QualityName
    measured_value: float


class PlantStateResponse(BaseModel):
    """Everything the physics engine publishes about the plant at one step."""

    timestamp_ms: int
    simulation: SimulationStatusResponse
    boiler: BoilerStateResponse
    turbine: TurbineStateResponse
    actuators: ActuatorStateResponse
    emissions: EmissionsResponse
    condenser: CondenserResponse
    health: EquipmentHealthResponse
    performance: PerformanceResponse
    faults: list[FaultResponse]
    sensors: list[SensorStatusResponse]


class ScenarioResponse(BaseModel):
    name: str
    title: str
    description: str


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioResponse]
    current: str


class SimulationAckResponse(BaseModel):
    accepted: bool
    reason: str
    timestamp_ms: int
    status: SimulationStatusResponse


FaultKindName = Literal[
    "burner_fouling",
    "steam_leak",
    "feedwater_pump_failure",
    "valve_stuck",
    "sensor_drift",
    "sensor_failure",
]


class SpeedRequest(BaseModel):
    speed_factor: float = Field(
        ..., gt=0.0, le=50.0, description="Simulated seconds per wall-clock second."
    )


class StepRequest(BaseModel):
    steps: int = Field(..., ge=1, le=3600, description="Only while paused.")


class ScenarioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern="^[a-z0-9_]+$")


class FaultRequest(BaseModel):
    kind: FaultKindName
    target: str = Field(
        default="",
        max_length=64,
        description="Valve (fuel, feedwater, steam, spray) or sensor id, if the kind needs one.",
    )
    severity: float = Field(default=1.0, ge=-1.0, le=1.0)
    ramp_s: float = Field(default=0.0, ge=0.0, le=3600.0)


class FaultAckResponse(BaseModel):
    accepted: bool
    reason: str
    timestamp_ms: int
    faults: list[FaultResponse]


class ScenarioRunResponse(BaseModel):
    id: int
    kind: Literal["scenario", "fault_injected", "fault_cleared"]
    scenario: str
    run_id: int
    fault_id: str | None
    fault_label: str | None
    severity: float | None
    simulation_time_s: float
    user_id: int | None
    username: str
    at_ms: int


class ScenarioRunPageResponse(BaseModel):
    items: list[ScenarioRunResponse]
    total: int
    limit: int
    offset: int
