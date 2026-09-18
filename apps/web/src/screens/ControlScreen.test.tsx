import { describe, expect, it } from "vitest";

import { FAULT_KINDS } from "./EngineerScreen";
import { LIMITS, inRange } from "./ControlScreen";
import { PASSWORD_MIN_LENGTH, USERNAME_PATTERN } from "./UsersScreen";

describe("command limits", () => {
  it("match the gateway's validation in display units", () => {
    expect(inRange(300, LIMITS.loadMw)).toBe(true);
    expect(inRange(300.1, LIMITS.loadMw)).toBe(false);
    expect(inRange(49.9, LIMITS.pressureBar)).toBe(false);
    expect(inRange(185, LIMITS.pressureBar)).toBe(true);
    expect(inRange(0.5, LIMITS.levelM)).toBe(true);
    // 848 K, the gateway's ceiling for the steam temperature setpoint.
    expect(inRange(574.85, LIMITS.steamTempC)).toBe(true);
    expect(inRange(575, LIMITS.steamTempC)).toBe(false);
  });

  it("refuse what is not a number", () => {
    expect(inRange(Number(""), LIMITS.valvePct)).toBe(true);
    expect(inRange(Number("abc"), LIMITS.valvePct)).toBe(false);
    expect(inRange(Number.POSITIVE_INFINITY, LIMITS.loadMw)).toBe(false);
  });
});

describe("fault kinds", () => {
  it("offer every kind the physics engine accepts, with its admissible severity", () => {
    expect(FAULT_KINDS.map((spec) => spec.kind).sort()).toEqual([
      "burner_fouling",
      "feedwater_pump_failure",
      "sensor_drift",
      "sensor_failure",
      "steam_leak",
      "valve_stuck",
    ]);
    for (const spec of FAULT_KINDS) {
      if (spec.severity) {
        expect(spec.severity.initial).toBeGreaterThanOrEqual(spec.severity.min);
        expect(spec.severity.initial).toBeLessThanOrEqual(spec.severity.max);
      }
    }
  });
});

describe("user administration", () => {
  it("follows the gateway's username pattern and password length", () => {
    expect(USERNAME_PATTERN.test("shift.operator")).toBe(true);
    expect(USERNAME_PATTERN.test("ab")).toBe(false);
    expect(USERNAME_PATTERN.test("with space")).toBe(false);
    expect(PASSWORD_MIN_LENGTH).toBe(12);
  });
});
