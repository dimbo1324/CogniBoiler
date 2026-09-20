// The backend speaks SI units only. Conversion for display happens here and nowhere else.

const PASCALS_PER_BAR = 1e5;
const PASCALS_PER_KILOPASCAL = 1e3;
const KELVIN_AT_ZERO_CELSIUS = 273.15;
const WATTS_PER_MEGAWATT = 1e6;
const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;
const KILOGRAMS_PER_TONNE = 1000;
const KILOJOULES_PER_KILOWATT_HOUR = 3600;
const JOULES_PER_MEGAWATT_HOUR = 3.6e9;

export function pascalsToBar(pascals: number): number {
  return pascals / PASCALS_PER_BAR;
}

export function barToPascals(bar: number): number {
  return bar * PASCALS_PER_BAR;
}

export function pascalsToKilopascals(pascals: number): number {
  return pascals / PASCALS_PER_KILOPASCAL;
}

export function kelvinToCelsius(kelvin: number): number {
  return kelvin - KELVIN_AT_ZERO_CELSIUS;
}

export function celsiusToKelvin(celsius: number): number {
  return celsius + KELVIN_AT_ZERO_CELSIUS;
}

export function wattsToMegawatts(watts: number): number {
  return watts / WATTS_PER_MEGAWATT;
}

export function megawattsToWatts(megawatts: number): number {
  return megawatts * WATTS_PER_MEGAWATT;
}

export function kilogramsPerSecondToTonnesPerHour(kilogramsPerSecond: number): number {
  return (kilogramsPerSecond * SECONDS_PER_HOUR) / KILOGRAMS_PER_TONNE;
}

export function pascalsPerSecondToBarPerMinute(pascalsPerSecond: number): number {
  return (pascalsPerSecond * SECONDS_PER_MINUTE) / PASCALS_PER_BAR;
}

export function fractionToPercent(fraction: number): number {
  return fraction * 100;
}

/** Heat rate from joules of heat per joule of electricity to kJ/kWh. */
export function heatRateToKilojoulesPerKilowattHour(joulesPerJoule: number): number {
  return joulesPerJoule * KILOJOULES_PER_KILOWATT_HOUR;
}

/** CO2 intensity from kg per joule of electricity to kg/MWh. */
export function co2PerJouleToKilogramsPerMegawattHour(kilogramsPerJoule: number): number {
  return kilogramsPerJoule * JOULES_PER_MEGAWATT_HOUR;
}

// Longest first: `pressure_rate_pa_s` ends with `_pa_s`, not with `_s`.
const UNIT_SUFFIXES: readonly (readonly [string, string])[] = [
  ["_pa_s", "Pa/s"],
  ["_kg_s", "kg/s"],
  ["_pa", "Pa"],
  ["_k", "K"],
  ["_m", "m"],
  ["_w", "W"],
  ["_pct", "%"],
];

export function formatReading(value: number | null | undefined, fractionDigits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(fractionDigits) : "—";
}

export interface DisplayQuantity {
  value: string;
  unit: string;
}

/**
 * A value in an SI unit, as the alarm and control-loop contracts name it, shown in the unit
 * an operator reads: bar, °C, MW, t/h, bar/min.
 */
export function displayQuantity(value: number | null | undefined, siUnit: string): DisplayQuantity {
  const finite = typeof value === "number" && Number.isFinite(value) ? value : Number.NaN;
  switch (siUnit) {
    case "Pa":
      return { value: formatReading(pascalsToBar(finite), 2), unit: "bar" };
    case "K":
      return { value: formatReading(kelvinToCelsius(finite), 1), unit: "°C" };
    case "W":
      return { value: formatReading(wattsToMegawatts(finite), 1), unit: "MW" };
    case "kg/s":
      return { value: formatReading(kilogramsPerSecondToTonnesPerHour(finite), 1), unit: "t/h" };
    case "Pa/s":
      return { value: formatReading(pascalsPerSecondToBarPerMinute(finite), 2), unit: "bar/min" };
    case "m":
      return { value: formatReading(finite, 2), unit: "m" };
    default:
      return { value: formatReading(finite, 2), unit: siUnit };
  }
}

/**
 * The SI unit a contract field is named after: `pressure_pa` is Pa, `pressure_rate_pa_s` is
 * Pa/s. The suffixes are part of the naming rule of the contracts (SI unit in the field
 * name), so a value can be shown correctly without a table of every parameter. An unknown
 * suffix yields no unit rather than a wrong one.
 */
export function siUnitOf(parameter: string): string {
  const match = UNIT_SUFFIXES.find(([suffix]) => parameter.endsWith(suffix));
  return match ? match[1] : "";
}

export function formatQuantity(value: number | null | undefined, siUnit: string): string {
  const quantity = displayQuantity(value, siUnit);
  return quantity.unit ? `${quantity.value} ${quantity.unit}` : quantity.value;
}

function pad(value: number, width = 2): string {
  return String(Math.trunc(Math.abs(value))).padStart(width, "0");
}

/** The UTC offset of a moment in this browser's zone, e.g. "UTC−03:00". */
export function utcOffsetLabel(date: Date): string {
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes < 0 ? "−" : "+";
  const minutes = Math.abs(offsetMinutes);
  return `UTC${sign}${pad(Math.floor(minutes / 60))}:${pad(minutes % 60)}`;
}

/** A moment with its date, seconds and zone: "2026-09-18 09:30:15 UTC−03:00". */
export function formatDateTime(timestampMs: number | null | undefined): string {
  if (typeof timestampMs !== "number" || !Number.isFinite(timestampMs) || timestampMs <= 0) {
    return "—";
  }
  const date = new Date(timestampMs);
  const day = `${String(date.getFullYear())}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return `${day} ${time} ${utcOffsetLabel(date)}`;
}

/** The value of a datetime-local input, in this browser's zone, as epoch milliseconds. */
export function localInputToMs(value: string): number | null {
  if (!value) {
    return null;
  }
  const ms = new Date(value).getTime();
  return Number.isFinite(ms) ? ms : null;
}

/** A duration in seconds as h:mm:ss, for simulated time. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  return `${String(hours)}:${pad(Math.floor((whole % 3600) / 60))}:${pad(whole % 60)}`;
}

/** A parameter name from the contracts ("water_level_m") as words ("water level"). */
export function parameterLabel(parameter: string): string {
  const known: Record<string, string> = {
    pressure_pa: "drum pressure",
    water_level_m: "drum level",
    water_temp_k: "drum water temperature",
    flue_gas_temp_k: "furnace gas temperature",
    steam_temp_k: "main steam temperature",
    pressure_rate_pa_s: "drum pressure rate",
  };
  const label = known[parameter];
  if (label) {
    return label;
  }
  return parameter
    .replace(/_(pa|k|m|w|kg_s|pa_s)$/u, "")
    .replace(/_quality$/u, " instrument")
    .replace(/_/gu, " ");
}
