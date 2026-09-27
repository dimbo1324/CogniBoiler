import { describe, expect, it } from "vitest";

import { plantState } from "../test/fixtures";
import { DEFAULT_TREND_IDS, TREND_PARAMETERS, trendParameter } from "./parameters";

describe("trend parameters", () => {
  it("have unique ids and the defaults exist", () => {
    const ids = TREND_PARAMETERS.map((parameter) => parameter.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of DEFAULT_TREND_IDS) {
      expect(trendParameter(id)).toBeDefined();
    }
    expect(trendParameter("nowhere")).toBeUndefined();
  });

  it("read a live snapshot and convert to the unit shown", () => {
    const state = plantState();
    for (const parameter of TREND_PARAMETERS) {
      const shown = parameter.display(parameter.live(state));
      expect(Number.isFinite(shown), parameter.id).toBe(true);
      expect(parameter.history.field.length).toBeGreaterThan(0);
    }
    const power = trendParameter("electrical_power");
    expect(power?.display(power.live(state))).toBeCloseTo(300.0, 1);
    expect(power?.unit).toBe("MW");
  });
});
