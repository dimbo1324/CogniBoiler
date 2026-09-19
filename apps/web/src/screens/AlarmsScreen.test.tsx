import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  acknowledgeAlarm,
  acknowledgeAllAlarms,
  fetchActiveAlarms,
  fetchAlarm,
  fetchAlarmHistory,
} from "../api/endpoints";
import { useCan } from "../session/SessionProvider";
import { alarm } from "../test/fixtures";
import { AlarmsScreen } from "./AlarmsScreen";

vi.mock("../api/endpoints", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/endpoints")>();
  return {
    ...original,
    fetchActiveAlarms: vi.fn(),
    fetchAlarm: vi.fn(),
    fetchAlarmHistory: vi.fn(),
    acknowledgeAlarm: vi.fn(),
    acknowledgeAllAlarms: vi.fn(),
  };
});

vi.mock("../session/SessionProvider", () => ({ useCan: vi.fn() }));

const critical = alarm({
  id: 7,
  alarm_id: "7",
  severity: "critical",
  parameter: "water_level_m",
  message: "water_level_m low critical: 0.9 m against limit 1 m",
  raised_at_ms: 1_789_700_000_000,
});
const warning = alarm({ id: 8, alarm_id: "8", raised_at_ms: 1_789_700_100_000 });
const acknowledged = alarm({
  id: 9,
  alarm_id: "9",
  state: "ACTIVE_ACK",
  acknowledged: true,
  acknowledged_by: "operator1",
  acknowledged_at_ms: 1_789_700_200_000,
  raised_at_ms: 1_789_700_200_000,
});

function renderScreen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <AlarmsScreen />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("active alarms", () => {
  beforeEach(() => {
    vi.mocked(useCan).mockReturnValue(true);
    vi.mocked(fetchActiveAlarms).mockResolvedValue([acknowledged, warning, critical]);
    vi.mocked(acknowledgeAlarm).mockResolvedValue({
      accepted: true,
      reason: "",
      timestamp_ms: 1,
      alarms: [],
    });
  });

  it("puts critical and unacknowledged alarms first and flashes the critical one", async () => {
    renderScreen();
    await screen.findByTestId("alarm-7");
    const rows = screen.getAllByTestId(/^alarm-/u).map((row) => row.dataset.testid);
    expect(rows).toEqual(["alarm-7", "alarm-8", "alarm-9"]);
    expect(screen.getByTestId("alarm-7").className).toContain("flashing");
    expect(screen.getByTestId("alarm-9").textContent).toContain("operator1");
    expect(screen.getByRole("button", { name: "Acknowledge all (2)" })).toBeDefined();
  });

  it("acknowledges one alarm and all alarms", async () => {
    vi.mocked(acknowledgeAllAlarms).mockResolvedValue({
      accepted: true,
      reason: "",
      timestamp_ms: 1,
      alarms: [],
    });
    renderScreen();
    const row = await screen.findByTestId("alarm-8");
    await userEvent.click(within(row).getByRole("button", { name: "Acknowledge" }));
    expect(vi.mocked(acknowledgeAlarm)).toHaveBeenCalledWith(8, undefined);
    await userEvent.click(screen.getByRole("button", { name: "Acknowledge all (2)" }));
    expect(vi.mocked(acknowledgeAllAlarms)).toHaveBeenCalledWith("");
    await waitFor(() => {
      expect(vi.mocked(fetchActiveAlarms).mock.calls.length).toBeGreaterThan(1);
    });
  });

  it("shows a refusal of the alarm service", async () => {
    vi.mocked(acknowledgeAlarm).mockResolvedValue({
      accepted: false,
      reason: "alarm is already acknowledged",
      timestamp_ms: 1,
      alarms: [],
    });
    renderScreen();
    const row = await screen.findByTestId("alarm-7");
    await userEvent.click(within(row).getByRole("button", { name: "Acknowledge" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Refused: alarm is already acknowledged",
    );
  });

  it("offers no acknowledgement to a viewer", async () => {
    vi.mocked(useCan).mockReturnValue(false);
    renderScreen();
    await screen.findByTestId("alarm-7");
    expect(screen.queryByRole("button", { name: /Acknowledge/u })).toBeNull();
    expect(screen.queryByText("Action")).toBeNull();
  });

  it("says when nothing is active", async () => {
    vi.mocked(fetchActiveAlarms).mockResolvedValue([]);
    renderScreen();
    expect(await screen.findByText("No active alarms.")).toBeDefined();
    expect(
      (screen.getByRole("button", { name: "Acknowledge all (0)" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("opens the transitions of a selected alarm", async () => {
    vi.mocked(fetchAlarm).mockResolvedValue({
      alarm: { ...critical, ack_comment: "checked the gauge glass", occurrence_count: 2 },
      transitions: [
        {
          id: 1,
          alarm_id: 7,
          from_state: null,
          to_state: "ACTIVE_UNACK",
          at_ms: 1_789_700_000_000,
          actor: "plc-controller",
          comment: null,
          value: 0.9,
        },
        {
          id: 2,
          alarm_id: 7,
          from_state: "ACTIVE_UNACK",
          to_state: "ACTIVE_ACK",
          at_ms: 1_789_700_050_000,
          actor: "operator1",
          comment: "checked the gauge glass",
          value: 0.9,
        },
      ],
    });
    renderScreen();
    const row = await screen.findByTestId("alarm-7");
    await userEvent.click(within(row).getByRole("button", { name: "drum level" }));
    const details = await screen.findByRole("region", { name: "Alarm details" });
    expect(details.textContent).toContain("Alarm 7: drum level (critical)");
    expect(details.textContent).toContain("checked the gauge glass");
    expect(within(details).getAllByRole("row")).toHaveLength(3);
    expect(vi.mocked(fetchAlarm).mock.calls[0]?.[0]).toBe(7);
  });
});

describe("alarm history", () => {
  beforeEach(() => {
    vi.mocked(useCan).mockReturnValue(true);
    vi.mocked(fetchActiveAlarms).mockResolvedValue([]);
    vi.mocked(fetchAlarmHistory).mockResolvedValue({
      items: [warning, acknowledged],
      total: 60,
      limit: 25,
      offset: 0,
    });
  });

  it("filters and pages the history", async () => {
    renderScreen();
    await userEvent.click(screen.getByRole("tab", { name: "History" }));
    await screen.findByText("1–25 of 60");
    const form = screen.getByRole("form", { name: "Alarm history filters" });
    await userEvent.selectOptions(within(form).getByLabelText("Severity"), "critical");
    await userEvent.type(within(form).getByLabelText("Parameter"), " water_level_m ");
    await userEvent.click(within(form).getByRole("button", { name: "Apply" }));
    await waitFor(() => {
      expect(vi.mocked(fetchAlarmHistory)).toHaveBeenLastCalledWith(
        {
          severity: "critical",
          parameter: "water_level_m",
          fromMs: null,
          toMs: null,
          limit: 25,
          offset: 0,
        },
        expect.anything(),
      );
    });
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => {
      expect(vi.mocked(fetchAlarmHistory).mock.lastCall?.[0].offset).toBe(25);
    });
    expect(screen.getByRole("tab", { name: "History" }).getAttribute("aria-selected")).toBe("true");
  });
});
