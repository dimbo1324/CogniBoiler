// The console's copies of gateway bounds against the committed OpenAPI schema, so a change of
// the gateway's validation fails here instead of drifting silently.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { LIMITS } from "../screens/ControlScreen";
import { FAULT_KINDS } from "../screens/EngineerScreen";
import { ROLES } from "../session/roles";
import { barToPascals, celsiusToKelvin, megawattsToWatts, percentToFraction } from "../units";
import { GATEWAY_LIMITS } from "./limits";

interface Property {
  minimum?: number;
  maximum?: number;
  minLength?: number;
  pattern?: string;
  enum?: string[];
  anyOf?: Property[];
}

interface OpenApi {
  components: { schemas: Record<string, { properties: Record<string, Property> }> };
  paths: Record<
    string,
    { get?: { parameters?: { name: string; schema: Property }[] } } | undefined
  >;
}

const schema = JSON.parse(
  readFileSync(
    resolve(import.meta.dirname, "../../../../shared/openapi/api-gateway.json"),
    "utf-8",
  ),
) as OpenApi;

function property(model: string, field: string): Property {
  const found = schema.components.schemas[model]?.properties[field];
  if (found === undefined) {
    throw new Error(`${model}.${field} is not in the OpenAPI schema`);
  }
  return found.anyOf?.[0] ?? found;
}

function bounds(model: string, field: string): [number | undefined, number | undefined] {
  const found = property(model, field);
  return [found.minimum, found.maximum];
}

describe("gateway limits", () => {
  it.each([
    ["LoadDemandRequest", "load_w", GATEWAY_LIMITS.loadW],
    ["SetpointRequest", "pressure_pa", GATEWAY_LIMITS.pressurePa],
    ["SetpointRequest", "water_level_m", GATEWAY_LIMITS.waterLevelM],
    ["SetpointRequest", "steam_temp_k", GATEWAY_LIMITS.steamTempK],
    ["ValveCommandRequest", "fuel_valve", GATEWAY_LIMITS.valveFraction],
    ["ValveCommandRequest", "feedwater_valve", GATEWAY_LIMITS.valveFraction],
    ["ValveCommandRequest", "steam_valve", GATEWAY_LIMITS.valveFraction],
    ["ValveCommandRequest", "spray_valve", GATEWAY_LIMITS.valveFraction],
    ["StepRequest", "steps", GATEWAY_LIMITS.steps],
    ["FaultRequest", "ramp_s", GATEWAY_LIMITS.faultRampS],
  ] as const)("%s.%s matches the schema", (model, field, range) => {
    expect(bounds(model, field)).toEqual([range[0], range[1]]);
  });

  it("match the password policy and the username pattern", () => {
    expect(property("UserCreateRequest", "password").minLength).toBe(
      GATEWAY_LIMITS.passwordMinLength,
    );
    expect(property("PasswordResetRequest", "new_password").minLength).toBe(
      GATEWAY_LIMITS.passwordMinLength,
    );
    expect(property("UserCreateRequest", "username").pattern).toBe(
      GATEWAY_LIMITS.usernamePattern.source,
    );
  });

  it("become the display ranges without leaving the gateway's", () => {
    expect(megawattsToWatts(LIMITS.loadMw[0])).toBe(GATEWAY_LIMITS.loadW[0]);
    expect(megawattsToWatts(LIMITS.loadMw[1])).toBe(GATEWAY_LIMITS.loadW[1]);
    expect(barToPascals(LIMITS.pressureBar[0])).toBeCloseTo(GATEWAY_LIMITS.pressurePa[0]);
    expect(barToPascals(LIMITS.pressureBar[1])).toBeCloseTo(GATEWAY_LIMITS.pressurePa[1]);
    expect(celsiusToKelvin(LIMITS.steamTempC[0])).toBeCloseTo(GATEWAY_LIMITS.steamTempK[0]);
    expect(celsiusToKelvin(LIMITS.steamTempC[1])).toBeCloseTo(GATEWAY_LIMITS.steamTempK[1]);
    expect(LIMITS.steamTempC).toEqual([126.85, 574.85]);
    expect(percentToFraction(LIMITS.valvePct[1])).toBe(GATEWAY_LIMITS.valveFraction[1]);
  });
});

describe("gateway enumerations", () => {
  it("list the same roles", () => {
    expect([...(property("UserCreateRequest", "role").enum ?? [])].sort()).toEqual(
      [...ROLES].sort(),
    );
  });

  it("offer every fault kind the gateway accepts", () => {
    expect([...(property("FaultRequest", "kind").enum ?? [])].sort()).toEqual(
      FAULT_KINDS.map((spec) => spec.kind).sort(),
    );
  });

  it("name the history measurements the gateway's pattern allows", () => {
    const parameter = schema.paths["/api/v1/history"]?.get?.parameters?.find(
      (item) => item.name === "measurement",
    );
    expect(parameter?.schema.pattern).toBe("^(boiler_sensors|turbine_sensors|plant_status)$");
  });
});
