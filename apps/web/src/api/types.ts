// Names for the gateway's REST shapes. Every one is an alias of the type generated from the
// committed OpenAPI schema, so a contract change reaches the console through the compiler.

import type { components } from "./schema.gen";

type Schemas = components["schemas"];

export type Role = "viewer" | "operator" | "engineer" | "admin";

export type TokenResponse = Schemas["TokenResponse"];
export type Profile = Schemas["ProfileResponse"];
export type MessageResponse = Schemas["MessageResponse"];

export type PlantState = Schemas["PlantStateResponse"];
export type SimulationStatus = Schemas["SimulationStatusResponse"];
export type Fault = Schemas["FaultResponse"];
export type PlcStatus = Schemas["PLCStatusResponse"];
export type PlcMode = PlcStatus["mode"];
export type AlarmCondition = Schemas["AlarmConditionResponse"];

export type Alarm = Schemas["AlarmResponse"];
export type AlarmState = Alarm["state"];
export type AlarmPage = Schemas["AlarmPageResponse"];
export type AlarmDetail = Schemas["AlarmDetailResponse"];
export type AcknowledgeResponse = Schemas["AcknowledgeResponse"];

export type HistoryMeasurement = "boiler_sensors" | "turbine_sensors" | "plant_status";
export type HistoryResponse = Schemas["HistoryResponse"];
export type Kpi = Schemas["KpiResponse"];

export type CommandAck = Schemas["CommandAckResponse"];
export type SetpointRequest = Schemas["SetpointRequest"];
export type ValveCommandRequest = Schemas["ValveCommandRequest"];

export type ScenarioList = Schemas["ScenarioListResponse"];
export type SimulationAck = Schemas["SimulationAckResponse"];
export type FaultRequest = Schemas["FaultRequest"];
export type FaultKind = FaultRequest["kind"];
export type FaultAck = Schemas["FaultAckResponse"];
export type ScenarioRunPage = Schemas["ScenarioRunPageResponse"];

export type AuditPage = Schemas["AuditPageResponse"];
export type AuditEntry = Schemas["AuditResponse"];

export type User = Schemas["UserResponse"];
export type UserPage = Schemas["UserPageResponse"];
export type UserCreateRequest = Schemas["UserCreateRequest"];
export type UserUpdateRequest = Schemas["UserUpdateRequest"];
export type SessionsRevoked = Schemas["SessionsRevokedResponse"];

export type Platform = Schemas["PlatformResponse"];
export type Readiness = Schemas["ReadinessResponse"];

// WebSocket /ws frames. The REST schema does not describe them; the gateway's
// routers/websocket.py does, and alarm changes and PLC events are the MQTT payloads of
// alert-manager and plc-controller passed through unchanged.

export type Channel = "telemetry" | "plc" | "alarms";

export interface PlcEvent {
  event_id: string;
  kind: string;
  source_service: string;
  operator_id: string;
  detail: Record<string, unknown>;
  timestamp_ms: number;
}

export interface AlarmChange {
  alarm: {
    id: number;
    key: string;
    severity: string;
    parameter: string;
    state: AlarmState;
    message: string;
  };
  transition: {
    id: number;
    alarm_id: number;
    from_state: AlarmState | null;
    to_state: AlarmState;
    at_ms: number;
    actor: string;
  };
  timestamp_ms: number;
}

export type DataFrame =
  | { type: "data"; channel: "telemetry"; kind: "state"; ts_ms: number; data: PlantState }
  | { type: "data"; channel: "plc"; kind: "status"; ts_ms: number; data: PlcStatus }
  | { type: "data"; channel: "plc"; kind: "event"; ts_ms: number; data: PlcEvent }
  | { type: "data"; channel: "alarms"; kind: "change"; ts_ms: number; data: AlarmChange };
