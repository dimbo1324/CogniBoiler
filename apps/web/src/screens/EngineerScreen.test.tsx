import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearAllFaults,
  clearFault,
  fetchScenarioRuns,
  fetchScenarios,
  injectFault,
  loadScenario,
  pauseSimulation,
  resumeSimulation,
  setSimulationSpeed,
  stepSimulation,
} from "../api/endpoints";
import type { Fault, PlantState, SimulationAck } from "../api/types";
import { useLive } from "../live/LiveProvider";
import type { LiveSnapshot } from "../live/store";
import { plantState } from "../test/fixtures";
import { EngineerScreen } from "./EngineerScreen";

vi.mock("../api/endpoints", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/endpoints")>();
  return {
    ...original,
    clearAllFaults: vi.fn(),
    clearFault: vi.fn(),
    fetchScenarioRuns: vi.fn(),
    fetchScenarios: vi.fn(),
    injectFault: vi.fn(),
    loadScenario: vi.fn(),
    pauseSimulation: vi.fn(),
    resumeSimulation: vi.fn(),
    setSimulationSpeed: vi.fn(),
    stepSimulation: vi.fn(),
  };
});

vi.mock("../live/LiveProvider", () => ({ useLive: vi.fn() }));

function fault(overrides: Partial<Fault> = {}): Fault {
  return {
    fault_id: "F0001",
    kind: "steam_leak",
    target: "",
    severity: 0.1,
    ramp_s: 0,
    started_at_s: 10,
    intensity: 0.5,
    label: "steam_leak",
    ...overrides,
  };
}

function live(plant: PlantState | null): LiveSnapshot {
  return {
    connection: "live",
    plant,
    plc: null,
    plantReceivedAtMs: null,
    events: [],
    lastAlarmChange: null,
    samplesVersion: 0,
  } as LiveSnapshot;
}

function paused(faults: Fault[] = []): PlantState {
  const state = plantState({ faults });
  return { ...state, simulation: { ...state.simulation, run_state: "paused", speed_factor: 10 } };
}

const accepted: SimulationAck = {
  accepted: true,
  reason: "",
  timestamp_ms: 1,
  status: paused().simulation,
};

function renderScreen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <EngineerScreen />
    </QueryClientProvider>,
  );
}

async function confirm(label: string) {
  await userEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: label }));
}

beforeEach(() => {
  vi.mocked(useLive).mockReturnValue(live(paused()));
  vi.mocked(fetchScenarios).mockResolvedValue({
    scenarios: [
      { name: "steady_state", title: "Nominal load", description: "250 MW" },
      { name: "hot_start", title: "Hot start", description: "Drum hot" },
    ],
    current: "steady_state",
  });
  vi.mocked(fetchScenarioRuns).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
  for (const action of [
    pauseSimulation,
    resumeSimulation,
    setSimulationSpeed,
    stepSimulation,
    loadScenario,
  ]) {
    vi.mocked(action).mockResolvedValue(accepted);
  }
  const faultAck = { accepted: true, reason: "", timestamp_ms: 1, faults: [] };
  vi.mocked(injectFault).mockResolvedValue(faultAck);
  vi.mocked(clearFault).mockResolvedValue(faultAck);
  vi.mocked(clearAllFaults).mockResolvedValue(faultAck);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("simulation", () => {
  it("waits for the plant before offering control", () => {
    vi.mocked(useLive).mockReturnValue(live(null));
    renderScreen();
    expect(screen.getByText("Waiting for the plant…")).toBeDefined();
  });

  it("resumes a paused run, sets the speed and steps", async () => {
    renderScreen();
    expect(screen.getByTestId("simulation-state").textContent).toBe("paused at 10×");
    await userEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(vi.mocked(resumeSimulation)).toHaveBeenCalledTimes(1);
    expect((await screen.findByRole("status")).textContent).toBe("Accepted.");

    await userEvent.selectOptions(screen.getByLabelText(/Speed/u), "50");
    await userEvent.click(screen.getByRole("button", { name: "Set speed" }));
    expect(vi.mocked(setSimulationSpeed)).toHaveBeenCalledWith(50);

    const steps = screen.getByLabelText(/Steps/u);
    await userEvent.clear(steps);
    await userEvent.type(steps, "60");
    await userEvent.click(screen.getByRole("button", { name: "Step" }));
    expect(vi.mocked(stepSimulation)).toHaveBeenCalledWith(60);
    expect(screen.getByRole("region", { name: "Last action" }).textContent).toContain("Step 60");
  });

  it("allows stepping only while paused and within 3600 steps", async () => {
    const running = paused();
    running.simulation.run_state = "running";
    vi.mocked(useLive).mockReturnValue(live(running));
    renderScreen();
    expect((screen.getByRole("button", { name: "Step" }) as HTMLButtonElement).disabled).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "Pause" }));
    expect(vi.mocked(pauseSimulation)).toHaveBeenCalledTimes(1);
    cleanup();
    vi.mocked(useLive).mockReturnValue(live(paused()));
    renderScreen();
    const steps = screen.getByLabelText(/Steps/u);
    await userEvent.clear(steps);
    await userEvent.type(steps, "3601");
    expect((screen.getByRole("button", { name: "Step" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows a refusal of the physics engine", async () => {
    vi.mocked(stepSimulation).mockResolvedValue({
      ...accepted,
      accepted: false,
      reason: "pause the simulation before stepping it",
    });
    renderScreen();
    await userEvent.click(screen.getByRole("button", { name: "Step" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Refused: pause the simulation before stepping it",
    );
  });
});

describe("scenarios", () => {
  it("loads a scenario after a confirmation and marks the current one", async () => {
    renderScreen();
    const table = await screen.findByRole("region", { name: "Scenarios" });
    const current = (await within(table).findByText("Nominal load")).closest("tr");
    expect(current?.textContent).toContain("current");
    const hot = within(table).getByText("Hot start").closest("tr") as HTMLElement;
    await userEvent.click(within(hot).getByRole("button", { name: "Load…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Load “Hot start”");
    await confirm("Load scenario");
    expect(vi.mocked(loadScenario)).toHaveBeenCalledWith("hot_start");
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });
});

describe("faults", () => {
  it("injects a fault without a target, with its severity and ramp", async () => {
    renderScreen();
    const section = screen.getByRole("region", { name: "Faults" });
    await userEvent.selectOptions(within(section).getByLabelText("Fault"), "steam_leak");
    const severity = within(section).getByLabelText(/Severity/u);
    expect((severity as HTMLInputElement).value).toBe("0.1");
    await userEvent.clear(severity);
    await userEvent.type(severity, "0.25");
    const ramp = within(section).getByLabelText("Develops over [s]");
    await userEvent.clear(ramp);
    await userEvent.type(ramp, "120");
    await userEvent.click(within(section).getByRole("button", { name: "Inject…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Steam leak, severity 0.25");
    await confirm("Inject fault");
    expect(vi.mocked(injectFault)).toHaveBeenCalledWith({
      kind: "steam_leak",
      target: "",
      severity: 0.25,
      ramp_s: 120,
    });
  });

  it("needs a target where the fault has one and keeps severity within limits", async () => {
    renderScreen();
    const section = screen.getByRole("region", { name: "Faults" });
    const inject = within(section).getByRole("button", { name: "Inject…" }) as HTMLButtonElement;
    await userEvent.selectOptions(within(section).getByLabelText("Fault"), "valve_stuck");
    expect(inject.disabled).toBe(true);
    await userEvent.selectOptions(within(section).getByLabelText("Valve"), "spray");
    expect(inject.disabled).toBe(false);
    await userEvent.click(inject);
    await confirm("Inject fault");
    expect(vi.mocked(injectFault)).toHaveBeenCalledWith({
      kind: "valve_stuck",
      target: "spray",
      severity: 1,
      ramp_s: 0,
    });

    await userEvent.selectOptions(within(section).getByLabelText("Fault"), "burner_fouling");
    const severity = within(section).getByLabelText(/Severity/u);
    await userEvent.clear(severity);
    await userEvent.type(severity, "0.6");
    expect(inject.disabled).toBe(true);
  });

  it("offers the instruments of the live plant as sensor targets", async () => {
    renderScreen();
    const section = screen.getByRole("region", { name: "Faults" });
    await userEvent.selectOptions(within(section).getByLabelText("Fault"), "sensor_failure");
    const options = within(within(section).getByLabelText("Sensor")).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "choose…",
      "drum pressure",
      "drum level",
      "electrical power",
    ]);
  });

  it("lists active faults and clears one or all", async () => {
    vi.mocked(useLive).mockReturnValue(
      live(
        paused([
          fault(),
          fault({
            fault_id: "F0002",
            kind: "valve_stuck",
            target: "spray",
            label: "valve_stuck:spray",
          }),
        ]),
      ),
    );
    renderScreen();
    const leak = screen.getByTestId("fault-steam_leak");
    expect(leak.textContent).toContain("50 %");
    await userEvent.click(within(leak).getByRole("button", { name: "Clear…" }));
    await confirm("Clear fault");
    expect(vi.mocked(clearFault)).toHaveBeenCalledWith("F0001");
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    await userEvent.click(screen.getByRole("button", { name: "Clear all…" }));
    await confirm("Clear all faults");
    expect(vi.mocked(clearAllFaults)).toHaveBeenCalledTimes(1);
  });

  it("says when no fault is active", () => {
    renderScreen();
    expect(screen.getByText("None.")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Clear all…" })).toBeNull();
  });
});

describe("run log", () => {
  it("lists scenario loads and fault changes with who made them", async () => {
    vi.mocked(fetchScenarioRuns).mockResolvedValue({
      items: [
        {
          id: 1,
          kind: "fault_injected",
          scenario: "steady_state",
          run_id: 1,
          fault_id: "F0001",
          fault_label: "steam_leak",
          severity: 0.1,
          simulation_time_s: 3725,
          user_id: 3,
          username: "engineer",
          at_ms: 1_789_700_000_000,
        },
      ],
      total: 1,
      limit: 20,
      offset: 0,
    });
    renderScreen();
    const log = screen.getByRole("region", { name: "Scenario and fault log" });
    const row = (await within(log).findByText("engineer")).closest("tr") as HTMLElement;
    expect(row.textContent).toContain("fault injected");
    expect(row.textContent).toContain("steam_leak");
    expect(row.textContent).toContain("1:02:05");
  });
});
