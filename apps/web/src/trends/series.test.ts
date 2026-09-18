import { describe, expect, it } from "vitest";

import type { HistoryResponse } from "../api/types";
import { plantState } from "../test/fixtures";
import { TREND_PARAMETERS, trendParameter, type TrendParameter } from "./parameters";
import { SampleBuffer, buildColumns, sampleOf } from "./series";

function parameter(id: string): TrendParameter {
  const found = trendParameter(id);
  if (found === undefined) {
    throw new Error(`unknown trend parameter ${id}`);
  }
  return found;
}

function history(
  measurement: HistoryResponse["measurement"],
  points: [number, Record<string, number>][],
): HistoryResponse {
  return {
    measurement,
    start_ms: 0,
    end_ms: 0,
    window_s: 1,
    points: points.map(([timestamp_ms, values]) => ({ measurement, timestamp_ms, values })),
  };
}

describe("buildColumns", () => {
  const power = parameter("electrical_power");
  const pressure = parameter("drum_pressure");

  it("joins measurements on their timestamps, in display units, with gaps", () => {
    const columns = buildColumns(
      [power, pressure],
      [
        history("turbine_sensors", [
          [1000, { electrical_power_w: 250e6 }],
          [2000, { electrical_power_w: 260e6 }],
        ]),
        history("boiler_sensors", [[2000, { pressure_pa: 140e5 }]]),
      ],
      [],
      0,
    );
    expect(columns.times).toEqual([1, 2]);
    expect(columns.values[0]).toEqual([250, 260]);
    expect(columns.values[1]).toEqual([null, 140]);
  });

  it("continues history with live samples newer than the last recorded point", () => {
    const live = [
      { timestampMs: 1500, values: { electrical_power: 1e6 } },
      { timestampMs: 3000, values: { electrical_power: 270e6 } },
    ];
    const columns = buildColumns(
      [power],
      [history("turbine_sensors", [[2000, { electrical_power_w: 260e6 }]])],
      live,
      0,
    );
    expect(columns.times).toEqual([2, 3]);
    expect(columns.values[0]).toEqual([260, 270]);
  });

  it("leaves out live samples older than the range", () => {
    const live = [
      { timestampMs: 500, values: { electrical_power: 200e6 } },
      { timestampMs: 5000, values: { electrical_power: 300e6 } },
    ];
    const columns = buildColumns([power], [], live, 1000);
    expect(columns.times).toEqual([5]);
    expect(columns.values[0]).toEqual([300]);
  });

  it("ignores values that are missing or not numbers", () => {
    const columns = buildColumns(
      [power],
      [history("turbine_sensors", [[1000, {}]])],
      [{ timestampMs: 2000, values: { electrical_power: Number.NaN } }],
      0,
    );
    expect(columns.values[0]).toEqual([null]);
  });
});

describe("sampleOf", () => {
  it("takes every trend parameter's SI value from a plant state", () => {
    const sample = sampleOf(plantState(), TREND_PARAMETERS);
    expect(sample.values.electrical_power).toBeCloseTo(300_000_574.22);
    expect(sample.values.drum_level).toBeCloseTo(4.7954);
    expect(Object.keys(sample.values)).toHaveLength(TREND_PARAMETERS.length);
  });
});

describe("SampleBuffer", () => {
  it("keeps the newest samples up to its capacity and drops repeats", () => {
    const buffer = new SampleBuffer(3);
    for (const timestampMs of [1, 2, 2, 3, 4]) {
      buffer.push({ timestampMs, values: {} });
    }
    expect(buffer.size).toBe(3);
    expect(buffer.since(0).map((sample) => sample.timestampMs)).toEqual([2, 3, 4]);
    expect(buffer.since(3).map((sample) => sample.timestampMs)).toEqual([3, 4]);
  });
});

describe("the trend catalogue", () => {
  it("names every parameter once and reads the historian's field names", () => {
    const ids = TREND_PARAMETERS.map((item) => item.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const item of TREND_PARAMETERS) {
      expect(item.history.field).toMatch(/^[a-z][a-z0-9_]{0,63}$/u);
    }
  });
});
