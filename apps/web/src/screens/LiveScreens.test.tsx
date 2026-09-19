import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useActiveAlarms } from "../alarms/queries";
import { fetchHistory, fetchKpi, fetchPlatform } from "../api/endpoints";
import type { Kpi, PlantState, Platform, PlcEvent, PlcStatus } from "../api/types";
import { tripDescription } from "../components/PlcPanel";
import { useLive, useLiveStore } from "../live/LiveProvider";
import { LiveStore, type LiveSnapshot } from "../live/store";
import { alarm, plantState, plcStatus } from "../test/fixtures";
import { TREND_PARAMETERS, trendParameter } from "../trends/parameters";
import { OverviewScreen } from "./OverviewScreen";
import { PlatformScreen } from "./PlatformScreen";
import { TrendsScreen, historyRequests } from "./TrendsScreen";

vi.mock("../api/endpoints", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/endpoints")>();
  return {
    ...original,
    fetchHistory: vi.fn(),
    fetchKpi: vi.fn(),
    fetchPlatform: vi.fn(),
  };
});

vi.mock("../alarms/queries", async (importOriginal) => {
  const original = await importOriginal<typeof import("../alarms/queries")>();
  return { ...original, useActiveAlarms: vi.fn() };
});

vi.mock("../live/LiveProvider", () => ({ useLive: vi.fn(), useLiveStore: vi.fn() }));

vi.mock("../components/TrendChart", () => ({
  TrendChart: ({ parameter, times }: { parameter: { label: string }; times: number[] }) => (
    <div data-testid="trend-chart">
      {parameter.label}: {times.length} points
    </div>
  ),
}));

function snapshot(overrides: Partial<LiveSnapshot> = {}): LiveSnapshot {
  return {
    connection: "live",
    plant: null,
    plc: null,
    plantReceivedAtMs: null,
    events: [],
    lastAlarmChange: null,
    samplesVersion: 0,
    ...overrides,
  };
}

function renderScreen(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{node}</MemoryRouter>
    </QueryClientProvider>,
  );
}

function mockAlarms(alarms = [alarm()], isError = false) {
  vi.mocked(useActiveAlarms).mockReturnValue({ data: alarms, isError } as ReturnType<
    typeof useActiveAlarms
  >);
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("OverviewScreen", () => {
  const event: PlcEvent = {
    event_id: "mode_changed:1",
    kind: "mode_changed",
    source_service: "plc-controller",
    operator_id: "operator1",
    detail: { from: "auto", to: "manual", nested: { ignored: true } },
    timestamp_ms: 1_789_700_000_000,
  };

  it("waits for the first plant state", () => {
    vi.mocked(useLive).mockReturnValue(snapshot());
    mockAlarms([]);
    renderScreen(<OverviewScreen />);
    expect(screen.getByRole("status").textContent).toBe("Waiting for the first plant state…");
    expect(screen.getByText("Waiting for the PLC…")).toBeDefined();
    expect(screen.getByText("No events since this page opened.")).toBeDefined();
    expect(screen.getByText("No active alarms.")).toBeDefined();
  });

  it("shows the unit, its faults and degraded instruments, events and alarms", () => {
    const plant: PlantState = plantState({
      faults: [
        {
          fault_id: "F1",
          kind: "sensor_drift",
          target: "drum_level",
          severity: 0.05,
          ramp_s: 0,
          started_at_s: 1,
          intensity: 1,
          label: "sensor_drift:drum_level",
        },
      ],
      sensors: [{ sensor_id: "drum_level", quality: "uncertain", measured_value: 4.9 }],
    });
    vi.mocked(useLive).mockReturnValue(
      snapshot({ plant, plc: plcStatus(), events: [event], plantReceivedAtMs: 1_789_700_000_000 }),
    );
    mockAlarms([
      alarm({ id: 1 }),
      alarm({ id: 2, severity: "critical", parameter: "pressure_pa", unit: "Pa", value: 190e5 }),
    ]);
    renderScreen(<OverviewScreen />);
    expect(screen.getByTestId("scenario").textContent).toBe("steady_state");
    expect(screen.getByTestId("active-faults").textContent).toBe("sensor_drift:drum_level");
    expect(screen.getByText("drum level: uncertain")).toBeDefined();
    const events = screen.getByRole("region", { name: "PLC events" });
    expect(events.textContent).toContain("mode changed by operator1 (from=auto, to=manual)");
    const alarms = within(screen.getByRole("region", { name: "Active alarms" })).getAllByRole(
      "listitem",
    );
    expect(alarms[0]?.className).toBe("error flashing");
    expect(alarms[0]?.textContent).toContain("critical drum pressure");
    expect(screen.getByTestId("plc-load-demand").textContent).toBe("300.0 MW");
  });

  it("says when the alarms cannot be read", () => {
    vi.mocked(useLive).mockReturnValue(snapshot());
    mockAlarms([], true);
    renderScreen(<OverviewScreen />);
    expect(screen.getByText("Alarms are unavailable.")).toBeDefined();
  });
});

describe("PLC trip description", () => {
  it("names the cause with its unit, a manual trip, or nothing", () => {
    const tripped: PlcStatus = plcStatus({
      mode: "estop",
      emergency_stop_active: true,
      trip_cause: { parameter: "water_level_m", value: 0.9, threshold: 1.0, timestamp_ms: 1 },
    });
    expect(tripDescription(tripped)).toBe("drum level 0.90 m (limit 1.00 m)");
    expect(tripDescription(plcStatus({ emergency_stop_active: true }))).toBe("Manual trip");
    expect(tripDescription(plcStatus())).toBeNull();
  });

  it("shows the blockers and when a reset is allowed", () => {
    vi.mocked(useLive).mockReturnValue(
      snapshot({
        plc: plcStatus({
          mode: "estop",
          emergency_stop_active: true,
          reset_blockers: ["drum level low"],
        }),
      }),
    );
    mockAlarms([]);
    renderScreen(<OverviewScreen />);
    const trip = within(screen.getByRole("region", { name: "PLC" })).getByRole("status");
    expect(trip.textContent).toContain("Tripped: Manual trip");
    expect(trip.textContent).toContain("Reset blocked: drum level low");
  });
});

describe("PlatformScreen", () => {
  const platform: Platform = {
    readiness: {
      status: "degraded",
      components: [
        { name: "database", state: "up", required: true, latency_ms: 1.25 },
        { name: "plc-controller", state: "down", required: false, latency_ms: null },
      ],
      checked_at_ms: 1_789_700_000_000,
    },
    telemetry_age_s: 0.42,
    websocket_clients: 3,
  };

  it("shows readiness, each component and the live data age", async () => {
    vi.mocked(useLive).mockReturnValue(snapshot({ plantReceivedAtMs: 1_789_700_000_000 }));
    vi.mocked(fetchPlatform).mockResolvedValue(platform);
    renderScreen(<PlatformScreen />);
    expect((await screen.findByTestId("platform-status")).textContent).toBe(
      "Degraded: the gateway serves, but a service is down.",
    );
    const down = screen.getByTestId("component-plc-controller");
    expect(down.className).toBe("severity-critical");
    expect(down.textContent).toContain("—");
    expect(screen.getByTestId("component-database").textContent).toContain("1.3 ms");
    const liveData = screen.getByRole("region", { name: "Live data" });
    expect(liveData.textContent).toContain("0.4 s ago");
    expect(liveData.textContent).toContain("3");
  });

  it("explains an unreachable gateway", async () => {
    vi.mocked(useLive).mockReturnValue(snapshot());
    vi.mocked(fetchPlatform).mockRejectedValue(new Error("Failed to fetch"));
    renderScreen(<PlatformScreen />);
    expect(await screen.findByText("Failed to fetch")).toBeDefined();
  });
});

const kpi: Kpi = {
  start_ms: 1_789_700_000_000,
  end_ms: 1_789_700_900_000,
  source: "raw",
  samples: 900,
  mean_electrical_power_w: 250e6,
  mean_fuel_heat_input_w: 625e6,
  net_efficiency: 0.4,
  boiler_efficiency: 0.9,
  turbine_heat_rate_j_per_j: 2.25,
  plant_heat_rate_j_per_j: 2.5,
  co2_intensity_kg_per_j: 1.5e-7,
  mean_nox_ppmv: 45,
  peak_nox_ppmv: 60,
  mean_health_pct: 99.5,
  lowest_health_pct: 99,
};

describe("TrendsScreen", () => {
  beforeEach(() => {
    vi.mocked(useLive).mockReturnValue(snapshot());
    vi.mocked(useLiveStore).mockReturnValue(new LiveStore());
    vi.mocked(fetchKpi).mockResolvedValue(kpi);
    vi.mocked(fetchHistory).mockResolvedValue({
      measurement: "turbine_sensors",
      start_ms: 0,
      end_ms: 1,
      window_s: 0,
      points: [],
    });
  });

  it("draws the default parameters live and reads KPIs for the last 15 minutes", async () => {
    renderScreen(<TrendsScreen />);
    expect(screen.getAllByTestId("trend-chart").length).toBeGreaterThan(0);
    const kpis = await screen.findByTestId("kpis");
    expect(kpis.textContent).toContain("250.0 MW");
    expect(kpis.textContent).toContain("40.00 %");
    expect(kpis.textContent).toContain("9000 kJ/kWh");
    expect(kpis.textContent).toContain("540 kg/MWh");
    const [startMs, endMs] = vi.mocked(fetchKpi).mock.calls[0] ?? [];
    expect((endMs ?? 0) - (startMs ?? 0)).toBe(15 * 60_000);
    expect(vi.mocked(fetchHistory)).not.toHaveBeenCalled();
  });

  it("reads recorded history for a fixed range, one request per measurement", async () => {
    renderScreen(<TrendsScreen />);
    await userEvent.click(screen.getByRole("button", { name: "1 h" }));
    await waitFor(() => {
      expect(vi.mocked(fetchHistory)).toHaveBeenCalled();
    });
    const measurements = vi.mocked(fetchHistory).mock.calls.map((call) => call[0]);
    expect(new Set(measurements).size).toBe(measurements.length);
    const [, , startMs, endMs, limit] = vi.mocked(fetchHistory).mock.calls[0] ?? [];
    expect((endMs ?? 0) - (startMs ?? 0)).toBe(60 * 60_000);
    expect(limit).toBe(600);
    expect(screen.getByRole("button", { name: "Reload history" })).toBeDefined();
    expect(screen.getByRole("button", { name: "1 h" }).getAttribute("aria-pressed")).toBe("true");
  });

  it("says when the historian does not answer", async () => {
    vi.mocked(fetchHistory).mockRejectedValue(new Error("down"));
    vi.mocked(fetchKpi).mockRejectedValue(new Error("down"));
    renderScreen(<TrendsScreen />);
    await userEvent.click(screen.getByRole("button", { name: "24 h" }));
    expect(
      await screen.findByText("History is unavailable: the historian did not answer.", {
        exact: false,
      }),
    ).toBeDefined();
    expect(
      await screen.findByText("KPIs are unavailable: the historian did not answer."),
    ).toBeDefined();
  });

  it("asks for a parameter when none is chosen", async () => {
    renderScreen(<TrendsScreen />);
    const picker = screen.getByRole("group", { name: "Parameters" });
    for (const box of within(picker).getAllByRole("checkbox")) {
      if ((box as HTMLInputElement).checked) {
        await userEvent.click(box);
      }
    }
    expect(screen.getByText("Choose at least one parameter.")).toBeDefined();
    expect(screen.queryAllByTestId("trend-chart")).toHaveLength(0);
  });

  it("groups history fields by measurement", () => {
    const chosen = ["electrical_power", "drum_pressure"]
      .map((id) => trendParameter(id))
      .filter((parameter) => parameter !== undefined);
    const requests = historyRequests(chosen);
    const all = historyRequests(TREND_PARAMETERS);
    expect(requests.flatMap((request) => request.fields)).toHaveLength(2);
    expect(all.flatMap((request) => request.fields)).toHaveLength(TREND_PARAMETERS.length);
    expect(new Set(all.map((request) => request.measurement)).size).toBe(all.length);
  });
});
