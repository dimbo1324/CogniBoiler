import { describe, expect, it } from "vitest";

import {
  barToPascals,
  celsiusToKelvin,
  co2PerJouleToKilogramsPerMegawattHour,
  displayQuantity,
  formatDateTime,
  formatDuration,
  formatQuantity,
  formatReading,
  fractionToPercent,
  heatRateToKilojoulesPerKilowattHour,
  kelvinToCelsius,
  kilogramsPerSecondToTonnesPerHour,
  megawattsToWatts,
  parameterLabel,
  pascalsPerSecondToBarPerMinute,
  pascalsToBar,
  pascalsToKilopascals,
  utcOffsetLabel,
  wattsToMegawatts,
} from "./units";

describe("unit conversion", () => {
  it("converts nominal drum pressure to bar and back", () => {
    expect(pascalsToBar(140e5)).toBeCloseTo(140);
    expect(barToPascals(140)).toBeCloseTo(140e5);
  });

  it("converts condenser pressure to kilopascals", () => {
    expect(pascalsToKilopascals(6042.8)).toBeCloseTo(6.0428);
  });

  it("converts steam temperature to degrees Celsius and back", () => {
    expect(kelvinToCelsius(825.65)).toBeCloseTo(552.5);
    expect(celsiusToKelvin(552.5)).toBeCloseTo(825.65);
  });

  it("converts electrical output to megawatts and back", () => {
    expect(wattsToMegawatts(250e6)).toBeCloseTo(250);
    expect(megawattsToWatts(300)).toBeCloseTo(300e6);
  });

  it("converts steam flow to tonnes per hour", () => {
    expect(kilogramsPerSecondToTonnesPerHour(250)).toBeCloseTo(900);
  });

  it("converts a pressure rate to bar per minute", () => {
    expect(pascalsPerSecondToBarPerMinute(1e5 / 60)).toBeCloseTo(1);
  });

  it("converts a fraction to a percentage", () => {
    expect(fractionToPercent(0.3672)).toBeCloseTo(36.72);
  });

  it("converts the heat rate to kJ/kWh: 2.7235 J/J is 9805 kJ/kWh", () => {
    expect(heatRateToKilojoulesPerKilowattHour(2.7235)).toBeCloseTo(9804.6);
  });

  it("converts CO2 per joule to kg/MWh", () => {
    expect(co2PerJouleToKilogramsPerMegawattHour(1.519e-7)).toBeCloseTo(546.84);
  });
});

describe("formatReading", () => {
  it("rounds to the requested precision", () => {
    expect(formatReading(139.96)).toBe("140.0");
    expect(formatReading(4.8123, 2)).toBe("4.81");
  });

  it("shows a dash instead of a number that is not finite or missing", () => {
    expect(formatReading(Number.NaN)).toBe("—");
    expect(formatReading(Number.POSITIVE_INFINITY)).toBe("—");
    expect(formatReading(null)).toBe("—");
    expect(formatReading(undefined)).toBe("—");
  });
});

describe("displayQuantity", () => {
  it("shows each SI unit of the contracts in the unit an operator reads", () => {
    expect(displayQuantity(140e5, "Pa")).toEqual({ value: "140.00", unit: "bar" });
    expect(displayQuantity(811, "K")).toEqual({ value: "537.9", unit: "°C" });
    expect(displayQuantity(300e6, "W")).toEqual({ value: "300.0", unit: "MW" });
    expect(displayQuantity(245, "kg/s")).toEqual({ value: "882.0", unit: "t/h" });
    expect(displayQuantity(-3000, "Pa/s")).toEqual({ value: "-1.80", unit: "bar/min" });
    expect(displayQuantity(4.8, "m")).toEqual({ value: "4.80", unit: "m" });
    expect(displayQuantity(0.5, "")).toEqual({ value: "0.50", unit: "" });
  });

  it("formats a quantity with its unit", () => {
    expect(formatQuantity(0.463, "m")).toBe("0.46 m");
    expect(formatQuantity(null, "Pa")).toBe("— bar");
  });
});

describe("time", () => {
  it("shows a moment with seconds and its UTC offset", () => {
    const text = formatDateTime(Date.UTC(2026, 8, 18, 12, 30, 15));
    expect(text).toMatch(/^2026-09-1[89] \d{2}:\d{2}:15 UTC[+−]\d{2}:\d{2}$/u);
  });

  it("labels the zone offset with its sign", () => {
    const west = { getTimezoneOffset: () => 180 } as Date;
    const east = { getTimezoneOffset: () => -330 } as Date;
    expect(utcOffsetLabel(west)).toBe("UTC−03:00");
    expect(utcOffsetLabel(east)).toBe("UTC+05:30");
  });

  it("shows nothing for a missing moment", () => {
    expect(formatDateTime(null)).toBe("—");
    expect(formatDateTime(0)).toBe("—");
  });

  it("formats simulated time as h:mm:ss", () => {
    expect(formatDuration(11_382)).toBe("3:09:42");
    expect(formatDuration(-1)).toBe("—");
  });
});

describe("parameterLabel", () => {
  it("names the protected parameters and instrument quality conditions", () => {
    expect(parameterLabel("water_level_m")).toBe("drum level");
    expect(parameterLabel("drum_level_quality")).toBe("drum level instrument");
    expect(parameterLabel("feedwater_flow_kg_s")).toBe("feedwater flow");
  });
});
