// Joins recorded history and the live stream into the columns a chart draws: one shared time
// axis, one column per parameter, in display units, with gaps where a value is missing.

import type { HistoryResponse, PlantState } from "../api/types";
import type { TrendParameter } from "./parameters";

export interface LiveSample {
  timestampMs: number;
  /** SI values by parameter id. */
  values: Readonly<Record<string, number>>;
}

export interface ChartColumns {
  /** Seconds since the epoch, ascending — the unit uPlot uses on its x axis. */
  times: number[];
  /** One column per parameter, aligned with `times`, in display units. */
  values: (number | null)[][];
}

export function sampleOf(state: PlantState, parameters: readonly TrendParameter[]): LiveSample {
  const values: Record<string, number> = {};
  for (const parameter of parameters) {
    values[parameter.id] = parameter.live(state);
  }
  return { timestampMs: state.timestamp_ms, values };
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * History first, then every live sample newer than the last recorded point and inside the
 * range. Points of different measurements are joined on their timestamps.
 */
export function buildColumns(
  parameters: readonly TrendParameter[],
  history: readonly HistoryResponse[],
  live: readonly LiveSample[],
  startMs: number,
): ChartColumns {
  const rows = new Map<number, (number | null)[]>();
  const rowAt = (timestampMs: number): (number | null)[] => {
    let row = rows.get(timestampMs);
    if (row === undefined) {
      row = parameters.map(() => null);
      rows.set(timestampMs, row);
    }
    return row;
  };

  let lastRecordedMs = Number.NEGATIVE_INFINITY;
  for (const response of history) {
    for (const point of response.points) {
      lastRecordedMs = Math.max(lastRecordedMs, point.timestamp_ms);
      parameters.forEach((parameter, index) => {
        if (parameter.history.measurement !== response.measurement) {
          return;
        }
        const value = finite(point.values?.[parameter.history.field]);
        if (value !== null) {
          rowAt(point.timestamp_ms)[index] = parameter.display(value);
        }
      });
    }
  }

  for (const sample of live) {
    if (sample.timestampMs <= lastRecordedMs || sample.timestampMs < startMs) {
      continue;
    }
    const row = rowAt(sample.timestampMs);
    parameters.forEach((parameter, index) => {
      const value = finite(sample.values[parameter.id]);
      row[index] = value === null ? null : parameter.display(value);
    });
  }

  const times = [...rows.keys()].sort((a, b) => a - b);
  return {
    times: times.map((ms) => ms / 1000),
    values: parameters.map((_, index) => times.map((ms) => rows.get(ms)?.[index] ?? null)),
  };
}

/** A ring of the most recent samples, oldest first. */
export class SampleBuffer {
  private readonly capacity: number;
  private items: LiveSample[] = [];

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  push(sample: LiveSample): void {
    const last = this.items[this.items.length - 1];
    if (last !== undefined && sample.timestampMs <= last.timestampMs) {
      // States are stamped with wall-clock time when published; one that is not newer is
      // the latest state replayed to a fresh subscription, already recorded.
      return;
    }
    this.items.push(sample);
    if (this.items.length > this.capacity) {
      this.items = this.items.slice(this.items.length - this.capacity);
    }
  }

  since(startMs: number): LiveSample[] {
    return this.items.filter((sample) => sample.timestampMs >= startMs);
  }

  get size(): number {
    return this.items.length;
  }
}
