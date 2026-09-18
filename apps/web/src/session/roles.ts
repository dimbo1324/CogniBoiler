// What each role may do in the console. This only decides what is shown: the gateway checks
// every request again, so hiding a control is a courtesy, not the protection.

import type { Role } from "../api/types";

const LEVEL: Record<Role, number> = { viewer: 1, operator: 2, engineer: 3, admin: 4 };

export function atLeast(role: Role | null | undefined, minimum: Role): boolean {
  return role !== null && role !== undefined && LEVEL[role] >= LEVEL[minimum];
}

export type Permission =
  | "acknowledge_alarms"
  | "set_load"
  | "set_mode"
  | "manual_valves"
  | "set_setpoints"
  | "reset_estop"
  | "control_simulation"
  | "read_audit"
  | "manage_users";

const MINIMUM: Record<Permission, Role> = {
  acknowledge_alarms: "operator",
  set_load: "operator",
  set_mode: "operator",
  manual_valves: "operator",
  set_setpoints: "engineer",
  reset_estop: "engineer",
  control_simulation: "engineer",
  read_audit: "admin",
  manage_users: "admin",
};

export function can(role: Role | null | undefined, permission: Permission): boolean {
  return atLeast(role, MINIMUM[permission]);
}
