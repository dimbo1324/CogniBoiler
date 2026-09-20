// The cache keys of TanStack Query, in one place.
//
// A mutation invalidates what it changed by naming the key, not by spelling the same
// array again somewhere else: `["alarms"]` was written in five files and `["simulation"]`
// in three, and a single typo in one of them would have left a screen showing stale data
// with nothing to show for it.

import type { AlarmHistoryFilter, AuditFilter } from "./endpoints";
import type { HistoryMeasurement } from "./types";

export const queryKeys = {
  alarms: {
    all: ["alarms"] as const,
    active: ["alarms", "active"] as const,
    detail: (alarmId: number) => ["alarms", "detail", alarmId] as const,
    history: (filter: AlarmHistoryFilter) => ["alarms", "history", filter] as const,
  },
  plc: {
    all: ["plc"] as const,
  },
  simulation: {
    all: ["simulation"] as const,
    scenarios: ["simulation", "scenarios"] as const,
    runs: (offset: number) => ["simulation", "runs", offset] as const,
  },
  users: {
    all: ["users"] as const,
    page: (offset: number) => ["users", offset] as const,
  },
  audit: (filter: AuditFilter) => ["audit", filter] as const,
  platform: ["platform"] as const,
  // The range is part of the key: a recorded range never changes, so it is cached until
  // the operator asks for a different one.
  history: (
    measurement: HistoryMeasurement,
    fields: readonly string[],
    range: string,
    anchorMs: number,
  ) => ["history", measurement, fields.join(","), range, anchorMs] as const,
  kpi: (range: string, anchorMs: number) => ["kpi", range, anchorMs] as const,
};
