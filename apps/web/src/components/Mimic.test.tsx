import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { plantState, plcStatus } from "../test/fixtures";
import { Mimic, equipmentInAlarm } from "./Mimic";

function condition(parameter: string, severity: "warning" | "critical") {
  return {
    key: `k:${parameter}:${severity}`,
    parameter,
    severity,
    direction: "low",
    value: 0,
    threshold: 0,
    message: "",
    since_ms: 0,
  };
}

describe("Mimic", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows live values in the units an operator reads", () => {
    render(<Mimic plant={plantState()} plc={plcStatus()} />);
    expect(screen.getByTestId("mimic-power").textContent).toContain("300.0");
    expect(screen.getByTestId("mimic-drum-pressure").textContent).toContain("140.0");
    expect(screen.getByTestId("mimic-drum-level").textContent).toContain("4.80");
    expect(screen.getByTestId("mimic-steam-temperature").textContent).toContain("537.8");
    expect(screen.getByTestId("mimic-steam-flow").textContent).toContain("882");
  });

  it("marks a value whose instrument is not good", () => {
    const plant = plantState({
      sensors: [{ sensor_id: "drum_level", quality: "bad", measured_value: 0 }],
    });
    render(<Mimic plant={plant} plc={plcStatus()} />);
    const level = screen.getByTestId("mimic-drum-level");
    expect(level.querySelector(".quality-bad")).not.toBeNull();
    expect(level.querySelector("title")?.textContent).toBe("Instrument quality: bad");
  });

  it("says when the unit is tripped", () => {
    render(
      <Mimic
        plant={plantState()}
        plc={plcStatus({ emergency_stop_active: true, mode: "estop" })}
      />,
    );
    expect(screen.getByText("E-STOP — fuel shut off")).toBeDefined();
  });
});

describe("equipmentInAlarm", () => {
  it("maps PLC conditions to equipment, critical over warning", () => {
    expect(
      equipmentInAlarm([
        condition("water_level_m", "warning"),
        condition("pressure_pa", "critical"),
        condition("steam_temp_k", "warning"),
        condition("unknown_parameter", "critical"),
      ]),
    ).toEqual({ drum: "critical", superheater: "warning" });
  });
});
