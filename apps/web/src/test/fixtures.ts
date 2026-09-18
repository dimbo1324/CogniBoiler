// Contract-shaped data for the unit tests: a plant state recorded from the running stack at
// 300 MW, a PLC status and alarms. Builders take overrides so a test states only what it needs.

import type { Alarm, PlantState, PlcStatus, TokenResponse } from "../api/types";

export function plantState(overrides: Partial<PlantState> = {}): PlantState {
  return {
    timestamp_ms: 1_789_735_692_392,
    simulation: {
      run_state: "running",
      speed_factor: 1,
      simulation_time_s: 11_382,
      step_count: 11_382,
      scenario: "steady_state",
      run_id: 1,
      step_s: 1,
    },
    boiler: {
      pressure_pa: 14_000_046.47,
      water_level_m: 4.7954,
      water_temp_k: 609.8751,
      flue_gas_temp_k: 1399.5993,
      internal_energy_j: 28_094_377_303.72,
      timestamp_ms: 1_789_735_692_392,
      quality: "good",
      fuel_flow_kg_s: 19.4538,
      feedwater_flow_kg_s: 206.5295,
      drum_steam_flow_kg_s: 235.3946,
      spray_flow_kg_s: 9.741,
      superheater_outlet_temp_k: 840.4539,
      economizer_outlet_temp_k: 529.9179,
      stack_temp_k: 445.2406,
      heat_release_w: 749_651_962.84,
      boiler_efficiency: 0.8378,
      feedwater_temp_k: 423.15,
      relief_flow_kg_s: 0,
    },
    turbine: {
      electrical_power_w: 300_000_574.22,
      shaft_power_w: 306_123_034.92,
      enthalpy_in_j_kg: 3_428_407.97,
      enthalpy_out_j_kg: 2_179_609.35,
      exhaust_pressure_pa: 6042.8099,
      steam_flow_kg_s: 245.1356,
      timestamp_ms: 1_789_735_692_392,
      steam_temp_in_k: 810.9965,
      exhaust_temp_k: 309.4315,
    },
    actuators: {
      fuel_valve_command: 0.7782,
      fuel_valve_position: 0.7782,
      feedwater_valve_command: 0.5219,
      feedwater_valve_position: 0.5435,
      steam_valve_command: 0.8453,
      steam_valve_position: 0.8453,
      spray_valve_command: 0.3247,
      spray_valve_position: 0.3247,
    },
    emissions: {
      co2_kg_s: 45.5802,
      nox_kg_s: 0.029,
      co_kg_s: 0.0058,
      nox_ppmv: 44.7612,
      co2_intensity_kg_per_mwh: 546.9618,
    },
    condenser: {
      backpressure_pa: 6042.8103,
      condensate_temp_k: 309.4402,
      cooling_water_temp_in_k: 288.15,
      cooling_water_temp_out_k: 302.992,
      heat_rejected_w: 497_030_178.27,
      loading: 0.71,
    },
    health: {
      turbine_hours: 3.1617,
      turbine_starts: 0,
      turbine_damage: 0,
      boiler_tube_hours: 3.1617,
      boiler_tube_damage: 0,
      pump_hours: 3.1617,
      overall_health_pct: 99.9977,
      maintenance_alarm: false,
      maintenance_critical: false,
    },
    performance: {
      fuel_heat_input_w: 817_059_360.04,
      heat_to_cycle_w: 684_541_442.79,
      net_efficiency: 0.3672,
      turbine_heat_rate_j_per_j: 2.2818,
      plant_heat_rate_j_per_j: 2.7235,
      co2_intensity_kg_per_j: 1.519e-7,
      boiler_efficiency: 0.8378,
      electrical_power_w: 300_000_574.22,
    },
    faults: [],
    sensors: [
      { sensor_id: "drum_pressure", quality: "good", measured_value: 14_000_046.47 },
      { sensor_id: "drum_level", quality: "good", measured_value: 4.7954 },
      { sensor_id: "electrical_power", quality: "good", measured_value: 300_000_574.22 },
    ],
    ...overrides,
  };
}

export function plcStatus(overrides: Partial<PlcStatus> = {}): PlcStatus {
  const setpoints = { pressure_pa: 14_000_000, water_level_m: 4.8, steam_temp_k: 811 };
  return {
    mode: "auto",
    emergency_stop_active: false,
    trip_cause: null,
    reset_permitted: false,
    reset_blockers: [],
    load_demand_w: 300e6,
    load_setpoint_w: 300e6,
    setpoints,
    active_setpoints: setpoints,
    latest_command: {
      fuel_valve: 0.778,
      feedwater_valve: 0.751,
      steam_valve: 0.845,
      spray_valve: 0.324,
      source: "pid",
      operator_id: "plc-auto",
      timestamp_ms: 1_789_734_633_896,
    },
    active_conditions: [],
    loops: [],
    warning_count: 0,
    trip_count: 0,
    run_id: 1,
    ...overrides,
  };
}

export function alarm(overrides: Partial<Alarm> = {}): Alarm {
  return {
    alarm_id: "1",
    id: 1,
    key: "plc-controller:water_level_m:low:warning",
    source_service: "plc-controller",
    severity: "warning",
    parameter: "water_level_m",
    direction: "low",
    unit: "m",
    state: "ACTIVE_UNACK",
    value: 1.9,
    threshold: 2,
    action: "warn",
    message: "water_level_m low warning: 1.9 m against limit 2 m",
    topic: "alerts/warning",
    occurred_at_ms: 1_789_700_000_000,
    raised_at_ms: 1_789_700_000_000,
    cleared_at_ms: null,
    acknowledged: false,
    acknowledged_at_ms: null,
    acknowledged_by: null,
    ack_comment: null,
    cleared: false,
    occurrence_count: 1,
    updated_at_ms: 1_789_700_000_000,
    ...overrides,
  };
}

export function tokens(overrides: Partial<TokenResponse> = {}): TokenResponse {
  return {
    access_token: "access-1",
    refresh_token: "refresh-1",
    token_type: "bearer",
    expires_in: 900,
    access_expires_at_ms: 1_000_900_000,
    session_expires_at_ms: 1_604_800_000,
    username: "operator",
    role: "operator",
    ...overrides,
  };
}

export function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

export function problem(status: number, code: string, detail = "refused"): Response {
  return jsonResponse(status, {
    type: `urn:cogniboiler:problem:${code}`,
    title: "Error",
    status,
    detail,
    code,
  });
}
