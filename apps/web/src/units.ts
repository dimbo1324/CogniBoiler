// The backend speaks SI units only. Conversion for display happens here and nowhere else.

const PASCALS_PER_BAR = 1e5;
const KELVIN_AT_ZERO_CELSIUS = 273.15;
const WATTS_PER_MEGAWATT = 1e6;
const SECONDS_PER_HOUR = 3600;
const KILOGRAMS_PER_TONNE = 1000;

export function pascalsToBar(pascals: number): number {
  return pascals / PASCALS_PER_BAR;
}

export function kelvinToCelsius(kelvin: number): number {
  return kelvin - KELVIN_AT_ZERO_CELSIUS;
}

export function wattsToMegawatts(watts: number): number {
  return watts / WATTS_PER_MEGAWATT;
}

export function kilogramsPerSecondToTonnesPerHour(kilogramsPerSecond: number): number {
  return (kilogramsPerSecond * SECONDS_PER_HOUR) / KILOGRAMS_PER_TONNE;
}

export function formatReading(value: number, fractionDigits = 1): string {
  return Number.isFinite(value) ? value.toFixed(fractionDigits) : "—";
}
