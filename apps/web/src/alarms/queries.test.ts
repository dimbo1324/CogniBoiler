import { describe, expect, it } from "vitest";

import { alarm } from "../test/fixtures";
import { annunciating, isUnacknowledged, sortAlarms } from "./queries";

describe("alarm ordering and annunciation", () => {
  const warning = alarm({ id: 1, severity: "warning", state: "ACTIVE_UNACK", raised_at_ms: 3 });
  const criticalAcked = alarm({
    id: 2,
    severity: "critical",
    state: "ACTIVE_ACK",
    raised_at_ms: 4,
  });
  const criticalNew = alarm({
    id: 3,
    severity: "critical",
    state: "ACTIVE_UNACK",
    raised_at_ms: 1,
  });
  const criticalClearedUnacked = alarm({
    id: 4,
    severity: "critical",
    state: "CLEARED_UNACK",
    raised_at_ms: 2,
  });

  it("counts an alarm that cleared before anyone acknowledged it as unacknowledged", () => {
    expect(isUnacknowledged(criticalClearedUnacked)).toBe(true);
    expect(isUnacknowledged(criticalAcked)).toBe(false);
  });

  it("annunciates only unacknowledged critical alarms", () => {
    expect(
      annunciating([warning, criticalAcked, criticalNew, criticalClearedUnacked]).map((a) => a.id),
    ).toEqual([3, 4]);
  });

  it("puts unacknowledged critical first, then acknowledged critical, then warnings", () => {
    expect(
      sortAlarms([warning, criticalAcked, criticalNew, criticalClearedUnacked]).map((a) => a.id),
    ).toEqual([4, 3, 2, 1]);
  });
});
