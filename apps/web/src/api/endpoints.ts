// One typed function per gateway route the console uses. SI units in, SI units out:
// conversion for display happens in units.ts.

import { request } from "./http";
import type {
  AcknowledgeResponse,
  Alarm,
  AlarmDetail,
  AlarmPage,
  AuditPage,
  CommandAck,
  FaultAck,
  FaultRequest,
  HistoryMeasurement,
  HistoryResponse,
  Kpi,
  MessageResponse,
  PlantState,
  Platform,
  PlcMode,
  PlcStatus,
  Profile,
  Role,
  ScenarioList,
  ScenarioRunPage,
  SessionsRevoked,
  SetpointRequest,
  SimulationAck,
  SimulationStatus,
  TokenResponse,
  User,
  UserCreateRequest,
  UserPage,
  UserUpdateRequest,
  ValveCommandRequest,
} from "./types";

// Sessions. The refresh token travels only in the httpOnly cookie the gateway sets on
// /auth; the console never reads or stores the copy in the response body.

export function signIn(username: string, password: string): Promise<TokenResponse> {
  return request("POST", "/auth/login", { body: { username, password }, auth: false });
}

export function refreshSession(): Promise<TokenResponse> {
  return request("POST", "/auth/refresh", { auth: false });
}

/** Closes the session of the refresh cookie; the gateway answers 200 in every case. */
export function signOut(): Promise<MessageResponse> {
  return request("POST", "/auth/logout", { auth: false });
}

export function fetchProfile(): Promise<Profile> {
  return request("GET", "/auth/me");
}

export function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<TokenResponse> {
  return request("POST", "/auth/password", {
    body: { current_password: currentPassword, new_password: newPassword },
  });
}

// Plant and PLC

export function fetchPlant(signal?: AbortSignal): Promise<PlantState> {
  return request("GET", "/api/v1/plant", { signal });
}

export function fetchPlcStatus(signal?: AbortSignal): Promise<PlcStatus> {
  return request("GET", "/api/v1/plc/status", { signal });
}

export function setLoadDemand(loadW: number): Promise<CommandAck> {
  return request("POST", "/api/v1/commands/load", { body: { load_w: loadW } });
}

export function setControlMode(mode: PlcMode): Promise<CommandAck> {
  return request("POST", "/api/v1/commands/mode", { body: { mode } });
}

export function sendValveCommand(command: ValveCommandRequest): Promise<CommandAck> {
  return request("POST", "/api/v1/commands/valve", { body: command });
}

export function updateSetpoints(setpoints: SetpointRequest): Promise<CommandAck> {
  return request("POST", "/api/v1/commands/setpoint", { body: setpoints });
}

export function resetEmergencyStop(): Promise<CommandAck> {
  return request("POST", "/api/v1/commands/reset", { body: {} });
}

// Alarms

export function fetchActiveAlarms(signal?: AbortSignal): Promise<Alarm[]> {
  return request("GET", "/api/v1/alarms", { query: { active_only: true, limit: 200 }, signal });
}

export interface AlarmHistoryFilter {
  severity?: "warning" | "critical" | "";
  parameter?: string;
  fromMs?: number | null;
  toMs?: number | null;
  limit: number;
  offset: number;
}

export function fetchAlarmHistory(
  filter: AlarmHistoryFilter,
  signal?: AbortSignal,
): Promise<AlarmPage> {
  return request("GET", "/api/v1/alarms/history", {
    query: {
      severity: filter.severity,
      parameter: filter.parameter,
      from_ms: filter.fromMs,
      to_ms: filter.toMs,
      limit: filter.limit,
      offset: filter.offset,
    },
    signal,
  });
}

export function fetchAlarm(alarmId: number, signal?: AbortSignal): Promise<AlarmDetail> {
  return request("GET", `/api/v1/alarms/${String(alarmId)}`, { signal });
}

export function acknowledgeAlarm(alarmId: number, comment = ""): Promise<AcknowledgeResponse> {
  return request("POST", `/api/v1/alarms/${String(alarmId)}/ack`, { body: { comment } });
}

export function acknowledgeAllAlarms(comment = ""): Promise<AcknowledgeResponse> {
  return request("POST", "/api/v1/alarms/ack-all", { body: { comment } });
}

// History and KPIs

export function fetchHistory(
  measurement: HistoryMeasurement,
  fields: readonly string[],
  startMs: number,
  endMs: number,
  limit: number,
  signal?: AbortSignal,
): Promise<HistoryResponse> {
  return request("GET", "/api/v1/history", {
    query: {
      measurement,
      fields: fields.join(","),
      start_ms: startMs,
      end_ms: endMs,
      limit,
    },
    signal,
  });
}

export function fetchKpi(startMs: number, endMs: number, signal?: AbortSignal): Promise<Kpi> {
  return request("GET", "/api/v1/kpi", { query: { start_ms: startMs, end_ms: endMs }, signal });
}

// Simulation (engineer)

export function fetchSimulation(signal?: AbortSignal): Promise<SimulationStatus> {
  return request("GET", "/api/v1/simulation", { signal });
}

export function fetchScenarios(signal?: AbortSignal): Promise<ScenarioList> {
  return request("GET", "/api/v1/simulation/scenarios", { signal });
}

export function pauseSimulation(): Promise<SimulationAck> {
  return request("POST", "/api/v1/simulation/pause");
}

export function resumeSimulation(): Promise<SimulationAck> {
  return request("POST", "/api/v1/simulation/resume");
}

export function setSimulationSpeed(speedFactor: number): Promise<SimulationAck> {
  return request("POST", "/api/v1/simulation/speed", { body: { speed_factor: speedFactor } });
}

export function stepSimulation(steps: number): Promise<SimulationAck> {
  return request("POST", "/api/v1/simulation/step", { body: { steps } });
}

export function loadScenario(name: string): Promise<SimulationAck> {
  return request("POST", "/api/v1/simulation/scenario", { body: { name } });
}

export function injectFault(fault: FaultRequest): Promise<FaultAck> {
  return request("POST", "/api/v1/simulation/faults", { body: fault });
}

export function clearFault(faultId: string): Promise<FaultAck> {
  return request("DELETE", `/api/v1/simulation/faults/${encodeURIComponent(faultId)}`);
}

export function clearAllFaults(): Promise<FaultAck> {
  return request("DELETE", "/api/v1/simulation/faults");
}

export function fetchScenarioRuns(
  limit: number,
  offset: number,
  signal?: AbortSignal,
): Promise<ScenarioRunPage> {
  return request("GET", "/api/v1/simulation/runs", { query: { limit, offset }, signal });
}

// Audit and users (admin)

export interface AuditFilter {
  username?: string;
  method?: string;
  endpoint?: string;
  minStatus?: number | null;
  fromMs?: number | null;
  toMs?: number | null;
  limit: number;
  offset: number;
}

export function fetchAudit(filter: AuditFilter, signal?: AbortSignal): Promise<AuditPage> {
  return request("GET", "/api/v1/audit", {
    query: {
      username: filter.username,
      method: filter.method,
      endpoint: filter.endpoint,
      min_status: filter.minStatus,
      from_ms: filter.fromMs,
      to_ms: filter.toMs,
      limit: filter.limit,
      offset: filter.offset,
    },
    signal,
  });
}

export function fetchUsers(limit: number, offset: number, signal?: AbortSignal): Promise<UserPage> {
  return request("GET", "/api/v1/users", { query: { limit, offset }, signal });
}

export function createUser(user: UserCreateRequest): Promise<User> {
  return request("POST", "/api/v1/users", { body: user });
}

export function updateUser(userId: number, change: UserUpdateRequest): Promise<User> {
  return request("PATCH", `/api/v1/users/${String(userId)}`, { body: change });
}

export function resetUserPassword(userId: number, newPassword: string): Promise<MessageResponse> {
  return request("POST", `/api/v1/users/${String(userId)}/password`, {
    body: { new_password: newPassword },
  });
}

export function revokeUserSessions(userId: number): Promise<SessionsRevoked> {
  return request("POST", `/api/v1/users/${String(userId)}/revoke-sessions`);
}

// Platform

export function fetchPlatform(signal?: AbortSignal): Promise<Platform> {
  return request("GET", "/api/v1/platform", { signal });
}

export const ROLES: readonly Role[] = ["viewer", "operator", "engineer", "admin"];
