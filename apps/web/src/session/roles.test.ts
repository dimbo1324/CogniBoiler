import { describe, expect, it } from "vitest";

import { ROLES, atLeast, can } from "./roles";

describe("roles", () => {
  it("lists the roles from least to most privileged", () => {
    expect(ROLES).toEqual(["viewer", "operator", "engineer", "admin"]);
  });

  it("orders viewer < operator < engineer < admin", () => {
    expect(atLeast("operator", "viewer")).toBe(true);
    expect(atLeast("operator", "engineer")).toBe(false);
    expect(atLeast("admin", "engineer")).toBe(true);
    expect(atLeast(null, "viewer")).toBe(false);
  });

  it("shows controls to the roles the gateway allows them to", () => {
    expect(can("viewer", "acknowledge_alarms")).toBe(false);
    expect(can("operator", "acknowledge_alarms")).toBe(true);
    expect(can("operator", "set_load")).toBe(true);
    expect(can("operator", "reset_estop")).toBe(false);
    expect(can("engineer", "reset_estop")).toBe(true);
    expect(can("engineer", "control_simulation")).toBe(true);
    expect(can("engineer", "manage_users")).toBe(false);
    expect(can("admin", "read_audit")).toBe(true);
  });
});
