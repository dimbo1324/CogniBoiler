import { describe, expect, it } from "vitest";

import {
  formatReading,
  kelvinToCelsius,
  kilogramsPerSecondToTonnesPerHour,
  pascalsToBar,
  wattsToMegawatts,
} from "./units";

describe("unit conversion", () => {
  it("converts nominal drum pressure to bar", () => {
    expect(pascalsToBar(140e5)).toBeCloseTo(140);
  });

  it("converts steam temperature to degrees Celsius", () => {
    expect(kelvinToCelsius(825.65)).toBeCloseTo(552.5);
  });

  it("converts electrical output to megawatts", () => {
    expect(wattsToMegawatts(250e6)).toBeCloseTo(250);
  });

  it("converts steam flow to tonnes per hour", () => {
    expect(kilogramsPerSecondToTonnesPerHour(250)).toBeCloseTo(900);
  });
});

describe("formatReading", () => {
  it("rounds to the requested precision", () => {
    expect(formatReading(139.96)).toBe("140.0");
    expect(formatReading(4.8123, 2)).toBe("4.81");
  });

  it("shows a dash instead of a number that is not finite", () => {
    expect(formatReading(Number.NaN)).toBe("—");
    expect(formatReading(Number.POSITIVE_INFINITY)).toBe("—");
  });
});
