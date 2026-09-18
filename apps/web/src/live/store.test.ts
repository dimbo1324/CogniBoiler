import { describe, expect, it, vi } from "vitest";

import type { PlcEvent } from "../api/types";
import { plantState, plcStatus } from "../test/fixtures";
import { LiveStore } from "./store";

function event(index: number): PlcEvent {
  return {
    event_id: `load_demand_changed:${String(index)}`,
    kind: "load_demand_changed",
    source_service: "plc-controller",
    operator_id: "operator",
    detail: { load_w: 300e6 },
    timestamp_ms: index,
  };
}

describe("LiveStore", () => {
  it("keeps the latest plant state and records a trend sample for each", () => {
    const store = new LiveStore(() => 42);
    const listener = vi.fn();
    store.subscribe(listener);

    store.apply({
      type: "data",
      channel: "telemetry",
      kind: "state",
      ts_ms: 1,
      data: plantState({ timestamp_ms: 10 }),
    });
    store.apply({
      type: "data",
      channel: "telemetry",
      kind: "state",
      ts_ms: 2,
      data: plantState({ timestamp_ms: 20 }),
    });

    expect(store.snapshot().plant?.timestamp_ms).toBe(20);
    expect(store.snapshot().plantReceivedAtMs).toBe(42);
    expect(store.snapshot().samplesVersion).toBe(2);
    expect(store.samples.size).toBe(2);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("keeps the PLC status and the newest fifty events, newest first", () => {
    const store = new LiveStore();
    store.apply({
      type: "data",
      channel: "plc",
      kind: "status",
      ts_ms: 1,
      data: plcStatus({ mode: "manual" }),
    });
    for (let index = 1; index <= 60; index += 1) {
      store.apply({
        type: "data",
        channel: "plc",
        kind: "event",
        ts_ms: index,
        data: event(index),
      });
    }
    expect(store.snapshot().plc?.mode).toBe("manual");
    expect(store.snapshot().events).toHaveLength(50);
    expect(store.snapshot().events[0]?.timestamp_ms).toBe(60);
  });

  it("reports the connection state", () => {
    const store = new LiveStore();
    store.setConnection("live");
    expect(store.snapshot().connection).toBe("live");
  });
});
