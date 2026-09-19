import { afterEach, describe, expect, it, vi } from "vitest";

import * as endpoints from "./endpoints";
import { request } from "./http";

vi.mock("./http", () => ({ request: vi.fn(() => Promise.resolve({})) }));

const mockedRequest = vi.mocked(request);
const signal = new AbortController().signal;

interface Case {
  name: string;
  call: () => Promise<unknown>;
  method: string;
  path: string;
  options?: Record<string, unknown>;
}

const cases: Case[] = [
  {
    name: "signIn",
    call: () => endpoints.signIn("operator", "pw"),
    method: "POST",
    path: "/auth/login",
    options: { body: { username: "operator", password: "pw" }, auth: false },
  },
  {
    name: "refreshSession",
    call: () => endpoints.refreshSession(),
    method: "POST",
    path: "/auth/refresh",
    options: { auth: false },
  },
  {
    name: "signOut",
    call: () => endpoints.signOut(),
    method: "POST",
    path: "/auth/logout",
    options: { auth: false },
  },
  { name: "fetchProfile", call: () => endpoints.fetchProfile(), method: "GET", path: "/auth/me" },
  {
    name: "fetchPlant",
    call: () => endpoints.fetchPlant(signal),
    method: "GET",
    path: "/api/v1/plant",
    options: { signal },
  },
  {
    name: "fetchPlcStatus",
    call: () => endpoints.fetchPlcStatus(signal),
    method: "GET",
    path: "/api/v1/plc/status",
    options: { signal },
  },
  {
    name: "setLoadDemand",
    call: () => endpoints.setLoadDemand(180e6),
    method: "POST",
    path: "/api/v1/commands/load",
    options: { body: { load_w: 180e6 } },
  },
  {
    name: "setControlMode",
    call: () => endpoints.setControlMode("manual"),
    method: "POST",
    path: "/api/v1/commands/mode",
    options: { body: { mode: "manual" } },
  },
  {
    name: "sendValveCommand",
    call: () =>
      endpoints.sendValveCommand({ fuel_valve: 0.1, feedwater_valve: 0.2, steam_valve: 0.3 }),
    method: "POST",
    path: "/api/v1/commands/valve",
    options: { body: { fuel_valve: 0.1, feedwater_valve: 0.2, steam_valve: 0.3 } },
  },
  {
    name: "updateSetpoints",
    call: () =>
      endpoints.updateSetpoints({ pressure_pa: 14e6, water_level_m: 4.8, steam_temp_k: 811 }),
    method: "POST",
    path: "/api/v1/commands/setpoint",
    options: { body: { pressure_pa: 14e6, water_level_m: 4.8, steam_temp_k: 811 } },
  },
  {
    name: "resetEmergencyStop",
    call: () => endpoints.resetEmergencyStop(),
    method: "POST",
    path: "/api/v1/commands/reset",
  },
  {
    name: "fetchActiveAlarms",
    call: () => endpoints.fetchActiveAlarms(signal),
    method: "GET",
    path: "/api/v1/alarms",
    options: { query: { active_only: true, limit: 200 }, signal },
  },
  {
    name: "fetchAlarmHistory",
    call: () =>
      endpoints.fetchAlarmHistory(
        { severity: "critical", parameter: "p", fromMs: 1, toMs: 2, limit: 10, offset: 20 },
        signal,
      ),
    method: "GET",
    path: "/api/v1/alarms/history",
    options: {
      query: {
        severity: "critical",
        parameter: "p",
        from_ms: 1,
        to_ms: 2,
        limit: 10,
        offset: 20,
      },
      signal,
    },
  },
  {
    name: "fetchAlarm",
    call: () => endpoints.fetchAlarm(7, signal),
    method: "GET",
    path: "/api/v1/alarms/7",
    options: { signal },
  },
  {
    name: "acknowledgeAlarm",
    call: () => endpoints.acknowledgeAlarm(7, "seen"),
    method: "POST",
    path: "/api/v1/alarms/7/ack",
    options: { body: { comment: "seen" } },
  },
  {
    name: "acknowledgeAllAlarms",
    call: () => endpoints.acknowledgeAllAlarms(),
    method: "POST",
    path: "/api/v1/alarms/ack-all",
    options: { body: { comment: "" } },
  },
  {
    name: "fetchHistory",
    call: () => endpoints.fetchHistory("boiler_sensors", ["a", "b"], 1, 2, 500, signal),
    method: "GET",
    path: "/api/v1/history",
    options: {
      query: { measurement: "boiler_sensors", fields: "a,b", start_ms: 1, end_ms: 2, limit: 500 },
      signal,
    },
  },
  {
    name: "fetchKpi",
    call: () => endpoints.fetchKpi(1, 2, signal),
    method: "GET",
    path: "/api/v1/kpi",
    options: { query: { start_ms: 1, end_ms: 2 }, signal },
  },
  {
    name: "fetchSimulation",
    call: () => endpoints.fetchSimulation(signal),
    method: "GET",
    path: "/api/v1/simulation",
    options: { signal },
  },
  {
    name: "fetchScenarios",
    call: () => endpoints.fetchScenarios(signal),
    method: "GET",
    path: "/api/v1/simulation/scenarios",
    options: { signal },
  },
  {
    name: "pauseSimulation",
    call: () => endpoints.pauseSimulation(),
    method: "POST",
    path: "/api/v1/simulation/pause",
  },
  {
    name: "resumeSimulation",
    call: () => endpoints.resumeSimulation(),
    method: "POST",
    path: "/api/v1/simulation/resume",
  },
  {
    name: "setSimulationSpeed",
    call: () => endpoints.setSimulationSpeed(10),
    method: "POST",
    path: "/api/v1/simulation/speed",
    options: { body: { speed_factor: 10 } },
  },
  {
    name: "stepSimulation",
    call: () => endpoints.stepSimulation(60),
    method: "POST",
    path: "/api/v1/simulation/step",
    options: { body: { steps: 60 } },
  },
  {
    name: "loadScenario",
    call: () => endpoints.loadScenario("hot_start"),
    method: "POST",
    path: "/api/v1/simulation/scenario",
    options: { body: { name: "hot_start" } },
  },
  {
    name: "injectFault",
    call: () =>
      endpoints.injectFault({ kind: "steam_leak", target: "", severity: 0.1, ramp_s: 30 }),
    method: "POST",
    path: "/api/v1/simulation/faults",
    options: { body: { kind: "steam_leak", target: "", severity: 0.1, ramp_s: 30 } },
  },
  {
    name: "clearFault",
    call: () => endpoints.clearFault("F 1/2"),
    method: "DELETE",
    path: "/api/v1/simulation/faults/F%201%2F2",
  },
  {
    name: "clearAllFaults",
    call: () => endpoints.clearAllFaults(),
    method: "DELETE",
    path: "/api/v1/simulation/faults",
  },
  {
    name: "fetchScenarioRuns",
    call: () => endpoints.fetchScenarioRuns(25, 50, signal),
    method: "GET",
    path: "/api/v1/simulation/runs",
    options: { query: { limit: 25, offset: 50 }, signal },
  },
  {
    name: "fetchAudit",
    call: () =>
      endpoints.fetchAudit(
        {
          username: "u",
          method: "POST",
          endpoint: "/api",
          minStatus: 400,
          fromMs: 1,
          toMs: 2,
          limit: 50,
          offset: 0,
        },
        signal,
      ),
    method: "GET",
    path: "/api/v1/audit",
    options: {
      query: {
        username: "u",
        method: "POST",
        endpoint: "/api",
        min_status: 400,
        from_ms: 1,
        to_ms: 2,
        limit: 50,
        offset: 0,
      },
      signal,
    },
  },
  {
    name: "fetchUsers",
    call: () => endpoints.fetchUsers(50, 100, signal),
    method: "GET",
    path: "/api/v1/users",
    options: { query: { limit: 50, offset: 100 }, signal },
  },
  {
    name: "createUser",
    call: () => endpoints.createUser({ username: "u1", password: "p", role: "viewer" }),
    method: "POST",
    path: "/api/v1/users",
    options: { body: { username: "u1", password: "p", role: "viewer" } },
  },
  {
    name: "updateUser",
    call: () => endpoints.updateUser(3, { is_active: false }),
    method: "PATCH",
    path: "/api/v1/users/3",
    options: { body: { is_active: false } },
  },
  {
    name: "resetUserPassword",
    call: () => endpoints.resetUserPassword(3, "new-password-1"),
    method: "POST",
    path: "/api/v1/users/3/password",
    options: { body: { new_password: "new-password-1" } },
  },
  {
    name: "revokeUserSessions",
    call: () => endpoints.revokeUserSessions(3),
    method: "POST",
    path: "/api/v1/users/3/revoke-sessions",
  },
  {
    name: "fetchPlatform",
    call: () => endpoints.fetchPlatform(signal),
    method: "GET",
    path: "/api/v1/platform",
    options: { signal },
  },
];

describe("gateway routes", () => {
  afterEach(() => {
    mockedRequest.mockClear();
  });

  it.each(cases)("$name sends $method $path", async ({ call, method, path, options }) => {
    await call();
    expect(mockedRequest).toHaveBeenCalledTimes(1);
    const [sentMethod, sentPath, sentOptions] = mockedRequest.mock.calls[0] ?? [];
    expect(sentMethod).toBe(method);
    expect(sentPath).toBe(path);
    if (options) {
      expect(sentOptions).toMatchObject(options);
    }
  });

  it("covers every exported request function", () => {
    const functions = Object.entries(endpoints)
      .filter(([, value]) => typeof value === "function")
      .map(([name]) => name)
      .filter((name) => name !== "changePassword");
    expect(functions.sort()).toEqual(cases.map((c) => c.name).sort());
  });

  it("changes the password with the current one", async () => {
    await endpoints.changePassword("old-password", "new-password-1");
    const [method, path, options] = mockedRequest.mock.calls[0] ?? [];
    expect([method, path]).toEqual(["POST", "/auth/password"]);
    expect(options).toMatchObject({
      body: { current_password: "old-password", new_password: "new-password-1" },
    });
  });

  it("lists the roles from least to most privileged", () => {
    expect(endpoints.ROLES).toEqual(["viewer", "operator", "engineer", "admin"]);
  });
});
