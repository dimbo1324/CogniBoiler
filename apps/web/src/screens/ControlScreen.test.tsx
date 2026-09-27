import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  resetEmergencyStop,
  sendValveCommand,
  setControlMode,
  setLoadDemand,
  updateSetpoints,
} from "../api/endpoints";
import { ApiError } from "../api/http";
import type { CommandAck, PlcStatus, Role } from "../api/types";
import { useLive } from "../live/LiveProvider";
import type { LiveSnapshot } from "../live/store";
import { plcStatus } from "../test/fixtures";
import { parseDecimal } from "../units";
import { ControlScreen, LIMITS, inRange } from "./ControlScreen";

vi.mock("../api/endpoints", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/endpoints")>();
  return {
    ...original,
    resetEmergencyStop: vi.fn(),
    sendValveCommand: vi.fn(),
    setControlMode: vi.fn(),
    setLoadDemand: vi.fn(),
    updateSetpoints: vi.fn(),
  };
});

vi.mock("../live/LiveProvider", () => ({ useLive: vi.fn() }));

const session = vi.hoisted(() => ({ role: "operator" as Role }));

vi.mock("../session/SessionProvider", async () => {
  const roles = await import("../session/roles");
  return {
    useCan: (permission: Parameters<typeof roles.can>[1]) => roles.can(session.role, permission),
  };
});

const accepted: CommandAck = { accepted: true, reason: "", timestamp_ms: 1 };

function live(plc: PlcStatus | null): LiveSnapshot {
  return {
    connection: "live",
    plant: null,
    plc,
    plantReceivedAtMs: null,
    events: [],
    lastAlarmChange: null,
    samplesVersion: 0,
  };
}

function renderScreen(role: Role, plc: PlcStatus | null = plcStatus()) {
  session.role = role;
  vi.mocked(useLive).mockReturnValue(live(plc));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ControlScreen />
    </QueryClientProvider>,
  );
}

function region(name: string): HTMLElement {
  return screen.getByRole("region", { name });
}

function button(scope: HTMLElement, name: string | RegExp): HTMLButtonElement {
  return within(scope).getByRole("button", { name }) as HTMLButtonElement;
}

async function retype(input: HTMLElement, text: string) {
  await userEvent.clear(input);
  if (text !== "") {
    await userEvent.type(input, text);
  }
}

async function confirm(label: string) {
  await userEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: label }));
}

beforeEach(() => {
  for (const command of [
    setLoadDemand,
    setControlMode,
    sendValveCommand,
    updateSetpoints,
    resetEmergencyStop,
  ]) {
    vi.mocked(command).mockResolvedValue(accepted);
  }
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("command limits", () => {
  it("match the gateway's validation in display units", () => {
    expect(inRange(300, LIMITS.loadMw)).toBe(true);
    expect(inRange(300.1, LIMITS.loadMw)).toBe(false);
    expect(inRange(49.9, LIMITS.pressureBar)).toBe(false);
    expect(inRange(185, LIMITS.pressureBar)).toBe(true);
    expect(inRange(0.5, LIMITS.levelM)).toBe(true);
    // 848 K, the gateway's ceiling for the steam temperature setpoint.
    expect(inRange(574.85, LIMITS.steamTempC)).toBe(true);
    expect(inRange(575, LIMITS.steamTempC)).toBe(false);
  });

  it("refuse what is not a number", () => {
    expect(inRange(parseDecimal(""), LIMITS.valvePct)).toBe(false);
    expect(inRange(parseDecimal("  "), LIMITS.valvePct)).toBe(false);
    expect(inRange(parseDecimal("abc"), LIMITS.valvePct)).toBe(false);
    expect(inRange(null, LIMITS.loadMw)).toBe(false);
    expect(inRange(Number.POSITIVE_INFINITY, LIMITS.loadMw)).toBe(false);
  });
});

describe("ControlScreen", () => {
  it("waits for the PLC before offering commands", () => {
    renderScreen("operator", null);
    expect(screen.getByRole("status").textContent).toBe("Waiting for the PLC…");
  });

  it("gives an operator load, mode and valves, but no setpoints or reset", () => {
    renderScreen("operator", plcStatus({ emergency_stop_active: true, mode: "estop" }));
    expect(region("Load")).toBeDefined();
    expect(region("Mode")).toBeDefined();
    expect(region("Manual valves")).toBeDefined();
    expect(screen.queryByRole("region", { name: "Setpoints" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Emergency stop reset" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Apply/u })).toBeNull();
  });

  it("gives an engineer the setpoints, and the reset only while tripped", () => {
    renderScreen("engineer");
    expect(region("Setpoints")).toBeDefined();
    expect(screen.queryByRole("region", { name: "Emergency stop reset" })).toBeNull();
    cleanup();
    renderScreen("engineer", plcStatus({ emergency_stop_active: true, mode: "estop" }));
    expect(region("Emergency stop reset")).toBeDefined();
  });

  it("hides every command from a viewer", () => {
    renderScreen("viewer");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("asks before changing the load, sends watts, and a cancel sends nothing", async () => {
    renderScreen("operator");
    const load = region("Load");
    await retype(within(load).getByLabelText(/New demand/u), "250");
    await userEvent.click(button(load, "Set load…"));
    expect(screen.getByRole("dialog").textContent).toContain("250.0 MW");
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(vi.mocked(setLoadDemand)).not.toHaveBeenCalled();

    await userEvent.click(button(load, "Set load…"));
    await confirm("Set 250.0 MW");
    expect(vi.mocked(setLoadDemand)).toHaveBeenCalledWith(250e6);
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    const last = region("Last command");
    expect(last.textContent).toContain("Change the load demand");
    expect(within(last).getByRole("status").textContent).toBe("Accepted.");
  });

  it("refuses a load outside the gateway's range before sending", async () => {
    renderScreen("operator");
    const load = region("Load");
    await retype(within(load).getByLabelText(/New demand/u), "301");
    expect(button(load, "Set load…").disabled).toBe(true);
    expect(load.textContent).toContain("0–300 MW");
  });

  it("never sends a cleared field as zero", async () => {
    renderScreen("engineer");
    const load = region("Load");
    await retype(within(load).getByLabelText(/New demand/u), "");
    expect(button(load, "Set load…").disabled).toBe(true);
    expect(load.textContent).toContain("0–300 MW");

    const valves = region("Manual valves");
    const feedwater = within(valves).getByLabelText(/Feedwater/u);
    await retype(feedwater, "");
    expect(feedwater.getAttribute("aria-invalid")).toBe("true");
    expect(button(valves, "Send positions…").disabled).toBe(true);
    await retype(feedwater, "12,5");
    expect(button(valves, "Send positions…").disabled).toBe(true);

    const setpoints = region("Setpoints");
    await retype(within(setpoints).getByLabelText(/Drum level/u), "");
    expect(button(setpoints, "Apply…").disabled).toBe(true);
    expect(vi.mocked(setLoadDemand)).not.toHaveBeenCalled();
    expect(vi.mocked(sendValveCommand)).not.toHaveBeenCalled();
  });

  it("switches the mode and trips the unit only after a confirmation", async () => {
    renderScreen("operator", plcStatus({ mode: "manual" }));
    const mode = region("Mode");
    expect(button(mode, "MANUAL…").disabled).toBe(true);
    await userEvent.click(button(mode, "AUTO…"));
    await confirm("Switch to AUTO");
    expect(vi.mocked(setControlMode)).toHaveBeenCalledWith("auto");
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    await userEvent.click(button(mode, "Emergency stop…"));
    expect(screen.getByRole("dialog", { name: "Trip the unit" })).toBeDefined();
    await confirm("Trip now");
    expect(vi.mocked(setControlMode)).toHaveBeenLastCalledWith("estop");
  });

  it("disables mode, trip and valve commands while the E-Stop is latched", () => {
    renderScreen("operator", plcStatus({ emergency_stop_active: true, mode: "estop" }));
    const mode = region("Mode");
    expect(button(mode, "AUTO…").disabled).toBe(true);
    expect(button(mode, "MANUAL…").disabled).toBe(true);
    expect(button(mode, "Emergency stop…").disabled).toBe(true);
    expect(button(region("Manual valves"), "Send positions…").disabled).toBe(true);
  });

  it("sends valve positions as fractions after a confirmation", async () => {
    renderScreen("operator");
    const valves = region("Manual valves");
    await retype(within(valves).getByLabelText(/Fuel/u), "50");
    await retype(within(valves).getByLabelText(/Feedwater/u), "40");
    await retype(within(valves).getByLabelText(/Turbine/u), "60");
    await retype(within(valves).getByLabelText(/Spray/u), "0");
    await userEvent.click(button(valves, "Send positions…"));
    expect(screen.getByRole("dialog").textContent).toContain(
      "Fuel 50.0 %, Feedwater 40.0 %, Turbine 60.0 %, Spray 0.0 %",
    );
    await confirm("Send and switch to MANUAL");
    expect(vi.mocked(sendValveCommand)).toHaveBeenCalledWith({
      fuel_valve: 0.5,
      feedwater_valve: 0.4,
      steam_valve: 0.6,
      spray_valve: 0,
    });
  });

  it("puts the current commands back into the valve fields", async () => {
    renderScreen("operator");
    const valves = region("Manual valves");
    const fuel = within(valves).getByLabelText(/Fuel/u) as HTMLInputElement;
    expect(fuel.value).toBe("77.8");
    await retype(fuel, "10");
    await userEvent.click(button(valves, "Current commands"));
    expect(fuel.value).toBe("77.8");
  });

  it("sends setpoints in pascals and kelvin", async () => {
    renderScreen("engineer");
    const setpoints = region("Setpoints");
    await retype(within(setpoints).getByLabelText(/Drum pressure/u), "160");
    await retype(within(setpoints).getByLabelText(/Drum level/u), "5");
    await retype(within(setpoints).getByLabelText(/Main steam temperature/u), "540");
    await userEvent.click(button(setpoints, "Apply…"));
    await confirm("Apply setpoints");
    expect(vi.mocked(updateSetpoints)).toHaveBeenCalledTimes(1);
    const sent = vi.mocked(updateSetpoints).mock.calls[0]?.[0];
    expect(sent?.pressure_pa).toBeCloseTo(16e6);
    expect(sent?.water_level_m).toBe(5);
    expect(sent?.steam_temp_k).toBeCloseTo(813.15);
  });

  it("resets the E-Stop after a confirmation and says why a reset is blocked", async () => {
    renderScreen(
      "engineer",
      plcStatus({
        emergency_stop_active: true,
        mode: "estop",
        reset_blockers: ["drum level below trip limit"],
      }),
    );
    const reset = region("Emergency stop reset");
    expect(reset.textContent).toContain("Reset is blocked: drum level below trip limit.");
    await userEvent.click(button(reset, "Reset E-Stop…"));
    await confirm("Reset E-Stop");
    expect(vi.mocked(resetEmergencyStop)).toHaveBeenCalledTimes(1);
  });

  it("shows the PLC's refusal of the last command", async () => {
    vi.mocked(setControlMode).mockResolvedValue({
      accepted: false,
      reason: "Emergency stop is active.",
      timestamp_ms: 1,
    });
    renderScreen("operator", plcStatus({ mode: "manual" }));
    await userEvent.click(button(region("Mode"), "AUTO…"));
    await confirm("Switch to AUTO");
    const last = await screen.findByRole("region", { name: "Last command" });
    await waitFor(() => {
      expect(within(last).getByRole("alert").textContent).toBe(
        "Refused: Emergency stop is active.",
      );
    });
  });

  it("shows the gateway's refusal of the last command", async () => {
    vi.mocked(setLoadDemand).mockRejectedValue(
      new ApiError({
        status: 403,
        code: "auth.forbidden",
        title: "Forbidden",
        detail: "Requires the operator role.",
        errors: [],
        retryAfterS: null,
      }),
    );
    renderScreen("operator");
    await userEvent.click(button(region("Load"), "Set load…"));
    await confirm("Set 300.0 MW");
    const last = await screen.findByRole("region", { name: "Last command" });
    await waitFor(() => {
      expect(within(last).getByRole("alert").textContent).toBe("Requires the operator role.");
    });
  });
});
