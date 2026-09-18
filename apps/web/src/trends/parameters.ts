// Every value the trends screen can draw: where it lives in a live plant snapshot, where the
// historian keeps it, and how it is shown. One table, so the live stream and the history of
// a parameter can never disagree on units.

import type { HistoryMeasurement, PlantState } from "../api/types";
import {
  fractionToPercent,
  kelvinToCelsius,
  kilogramsPerSecondToTonnesPerHour,
  pascalsToBar,
  pascalsToKilopascals,
  wattsToMegawatts,
} from "../units";

export interface TrendParameter {
  id: string;
  label: string;
  unit: string;
  digits: number;
  history: { measurement: HistoryMeasurement; field: string };
  /** The SI value in a live snapshot. */
  live: (state: PlantState) => number;
  /** SI to the unit shown. */
  display: (value: number) => number;
}

const same = (value: number): number => value;

export const TREND_PARAMETERS: readonly TrendParameter[] = [
  {
    id: "electrical_power",
    label: "Electrical output",
    unit: "MW",
    digits: 1,
    history: { measurement: "turbine_sensors", field: "electrical_power_w" },
    live: (s) => s.turbine.electrical_power_w,
    display: wattsToMegawatts,
  },
  {
    id: "drum_pressure",
    label: "Drum pressure",
    unit: "bar",
    digits: 2,
    history: { measurement: "boiler_sensors", field: "pressure_pa" },
    live: (s) => s.boiler.pressure_pa,
    display: pascalsToBar,
  },
  {
    id: "drum_level",
    label: "Drum level",
    unit: "m",
    digits: 3,
    history: { measurement: "boiler_sensors", field: "water_level_m" },
    live: (s) => s.boiler.water_level_m,
    display: same,
  },
  {
    id: "steam_temperature",
    label: "Main steam temperature",
    unit: "°C",
    digits: 1,
    history: { measurement: "turbine_sensors", field: "steam_temp_in_k" },
    live: (s) => s.turbine.steam_temp_in_k,
    display: kelvinToCelsius,
  },
  {
    id: "steam_flow",
    label: "Main steam flow",
    unit: "t/h",
    digits: 0,
    history: { measurement: "turbine_sensors", field: "steam_flow_kg_s" },
    live: (s) => s.turbine.steam_flow_kg_s,
    display: kilogramsPerSecondToTonnesPerHour,
  },
  {
    id: "feedwater_flow",
    label: "Feedwater flow",
    unit: "t/h",
    digits: 0,
    history: { measurement: "boiler_sensors", field: "feedwater_flow_kg_s" },
    live: (s) => s.boiler.feedwater_flow_kg_s,
    display: kilogramsPerSecondToTonnesPerHour,
  },
  {
    id: "fuel_flow",
    label: "Fuel flow",
    unit: "t/h",
    digits: 1,
    history: { measurement: "boiler_sensors", field: "fuel_flow_kg_s" },
    live: (s) => s.boiler.fuel_flow_kg_s,
    display: kilogramsPerSecondToTonnesPerHour,
  },
  {
    id: "furnace_gas_temperature",
    label: "Furnace exit gas temperature",
    unit: "°C",
    digits: 0,
    history: { measurement: "boiler_sensors", field: "flue_gas_temp_k" },
    live: (s) => s.boiler.flue_gas_temp_k,
    display: kelvinToCelsius,
  },
  {
    id: "stack_temperature",
    label: "Stack temperature",
    unit: "°C",
    digits: 1,
    history: { measurement: "boiler_sensors", field: "stack_temp_k" },
    live: (s) => s.boiler.stack_temp_k,
    display: kelvinToCelsius,
  },
  {
    id: "net_efficiency",
    label: "Net efficiency",
    unit: "%",
    digits: 2,
    history: { measurement: "plant_status", field: "net_efficiency" },
    live: (s) => s.performance.net_efficiency,
    display: fractionToPercent,
  },
  {
    id: "nox",
    label: "NOx",
    unit: "ppmv",
    digits: 1,
    history: { measurement: "plant_status", field: "nox_ppmv" },
    live: (s) => s.emissions.nox_ppmv,
    display: same,
  },
  {
    id: "co2_intensity",
    label: "CO2 intensity",
    unit: "kg/MWh",
    digits: 0,
    history: { measurement: "plant_status", field: "co2_intensity_kg_per_mwh" },
    live: (s) => s.emissions.co2_intensity_kg_per_mwh,
    display: same,
  },
  {
    id: "condenser_pressure",
    label: "Condenser pressure",
    unit: "kPa",
    digits: 2,
    history: { measurement: "plant_status", field: "condenser_backpressure_pa" },
    live: (s) => s.condenser.backpressure_pa,
    display: pascalsToKilopascals,
  },
  {
    id: "fuel_valve",
    label: "Fuel valve position",
    unit: "%",
    digits: 1,
    history: { measurement: "plant_status", field: "fuel_valve_position" },
    live: (s) => s.actuators.fuel_valve_position,
    display: fractionToPercent,
  },
  {
    id: "feedwater_valve",
    label: "Feedwater valve position",
    unit: "%",
    digits: 1,
    history: { measurement: "plant_status", field: "feedwater_valve_position" },
    live: (s) => s.actuators.feedwater_valve_position,
    display: fractionToPercent,
  },
  {
    id: "steam_valve",
    label: "Turbine valve position",
    unit: "%",
    digits: 1,
    history: { measurement: "plant_status", field: "steam_valve_position" },
    live: (s) => s.actuators.steam_valve_position,
    display: fractionToPercent,
  },
  {
    id: "spray_valve",
    label: "Spray valve position",
    unit: "%",
    digits: 1,
    history: { measurement: "plant_status", field: "spray_valve_position" },
    live: (s) => s.actuators.spray_valve_position,
    display: fractionToPercent,
  },
  {
    id: "health",
    label: "Equipment health",
    unit: "%",
    digits: 1,
    history: { measurement: "plant_status", field: "overall_health_pct" },
    live: (s) => s.health.overall_health_pct,
    display: same,
  },
];

const BY_ID = new Map(TREND_PARAMETERS.map((parameter) => [parameter.id, parameter]));

export function trendParameter(id: string): TrendParameter | undefined {
  return BY_ID.get(id);
}

export const DEFAULT_TREND_IDS: readonly string[] = [
  "electrical_power",
  "drum_pressure",
  "drum_level",
  "steam_temperature",
];
