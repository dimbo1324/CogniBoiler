// Live plant state as the console sees it: the latest snapshot of each channel, recent PLC
// events and a buffer of samples for the trends. Framework-free; React reads it through
// useSyncExternalStore.

import type { ConnectionState } from "../api/realtime";
import type { AlarmChange, DataFrame, PlantState, PlcEvent, PlcStatus } from "../api/types";
import { TREND_PARAMETERS } from "../trends/parameters";
import { SampleBuffer, sampleOf } from "../trends/series";

export interface LiveSnapshot {
  connection: ConnectionState;
  plant: PlantState | null;
  plc: PlcStatus | null;
  /** Browser time when the last plant state arrived [ms]. */
  plantReceivedAtMs: number | null;
  /** Newest first. */
  events: readonly PlcEvent[];
  lastAlarmChange: AlarmChange | null;
  /** Grows with every sample added to the trend buffer. */
  samplesVersion: number;
}

const MAX_EVENTS = 50;
// Fifteen minutes at the two samples per second the console subscribes to.
export const TREND_BUFFER_SAMPLES = 1800;

export class LiveStore {
  private current: LiveSnapshot = {
    connection: "connecting",
    plant: null,
    plc: null,
    plantReceivedAtMs: null,
    events: [],
    lastAlarmChange: null,
    samplesVersion: 0,
  };
  private readonly listeners = new Set<() => void>();
  readonly samples = new SampleBuffer(TREND_BUFFER_SAMPLES);
  private readonly now: () => number;

  constructor(now: () => number = () => Date.now()) {
    this.now = now;
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  readonly snapshot = (): LiveSnapshot => this.current;

  setConnection(connection: ConnectionState): void {
    this.update({ connection });
  }

  apply(frame: DataFrame): void {
    switch (frame.channel) {
      case "telemetry":
        this.samples.push(sampleOf(frame.data, TREND_PARAMETERS));
        this.update({
          plant: frame.data,
          plantReceivedAtMs: this.now(),
          samplesVersion: this.current.samplesVersion + 1,
        });
        return;
      case "plc":
        if (frame.kind === "status") {
          this.update({ plc: frame.data });
        } else {
          const events = [frame.data, ...this.current.events].slice(0, MAX_EVENTS);
          this.update({ events });
        }
        return;
      case "alarms":
        this.update({ lastAlarmChange: frame.data });
        return;
    }
  }

  private update(change: Partial<LiveSnapshot>): void {
    this.current = { ...this.current, ...change };
    for (const listener of this.listeners) {
      listener();
    }
  }
}
