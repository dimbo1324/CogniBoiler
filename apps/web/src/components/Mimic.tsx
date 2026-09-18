import type { AlarmCondition, PlantState, PlcStatus } from "../api/types";
import {
  formatReading,
  fractionToPercent,
  kelvinToCelsius,
  kilogramsPerSecondToTonnesPerHour,
  pascalsToBar,
  pascalsToKilopascals,
  wattsToMegawatts,
} from "../units";

// The drum level instrument spans 0–8 m; the trips sit at 0.5 m and 7.8 m.
const DRUM_LEVEL_SPAN_M = 8;
const DRUM = { x: 70, y: 50, width: 260, height: 60 };

type Equipment = "drum" | "furnace" | "superheater" | "generator" | "feedwater";

// Which piece of equipment an alarm condition of the PLC belongs to.
const EQUIPMENT_OF_PARAMETER: Record<string, Equipment> = {
  pressure_pa: "drum",
  pressure_rate_pa_s: "drum",
  water_level_m: "drum",
  water_temp_k: "drum",
  drum_pressure_quality: "drum",
  drum_level_quality: "drum",
  drum_water_temp_quality: "drum",
  flue_gas_temp_k: "furnace",
  furnace_gas_temp_quality: "furnace",
  fuel_flow_quality: "furnace",
  steam_temp_k: "superheater",
  steam_temp_quality: "superheater",
  steam_flow_quality: "superheater",
  electrical_power_quality: "generator",
  feedwater_flow_quality: "feedwater",
};

export function equipmentInAlarm(
  conditions: readonly AlarmCondition[],
): Partial<Record<Equipment, "warning" | "critical">> {
  const result: Partial<Record<Equipment, "warning" | "critical">> = {};
  for (const condition of conditions) {
    const equipment = EQUIPMENT_OF_PARAMETER[condition.parameter];
    if (equipment === undefined) {
      continue;
    }
    if (condition.severity === "critical" || result[equipment] === undefined) {
      result[equipment] = condition.severity === "critical" ? "critical" : "warning";
    }
  }
  return result;
}

function qualityOf(state: PlantState, sensorId: string): string {
  return state.sensors.find((sensor) => sensor.sensor_id === sensorId)?.quality ?? "good";
}

function Reading({
  x,
  y,
  label,
  value,
  unit,
  quality = "good",
  anchor = "start",
  testId,
}: {
  x: number;
  y: number;
  label: string;
  value: string;
  unit: string;
  quality?: string;
  anchor?: "start" | "middle" | "end";
  testId?: string;
}) {
  return (
    <text x={x} y={y} textAnchor={anchor} data-testid={testId}>
      {quality !== "good" && <title>{`Instrument quality: ${quality}`}</title>}
      <tspan className="label">{label} </tspan>
      <tspan className={`value quality-${quality}`}>{value}</tspan>
      <tspan className="label"> {unit}</tspan>
    </text>
  );
}

function Valve({
  x,
  y,
  position,
  vertical = false,
}: {
  x: number;
  y: number;
  position: number;
  vertical?: boolean;
}) {
  const open = Math.min(Math.max(position, 0), 1);
  const points = vertical
    ? `${String(x - 9)},${String(y - 9)} ${String(x + 9)},${String(y - 9)} ${String(x - 9)},${String(y + 9)} ${String(x + 9)},${String(y + 9)}`
    : `${String(x - 9)},${String(y - 9)} ${String(x + 9)},${String(y + 9)} ${String(x + 9)},${String(y - 9)} ${String(x - 9)},${String(y + 9)}`;
  return (
    <polygon
      points={points}
      className="valve"
      style={{
        fill: `color-mix(in srgb, var(--ok) ${String(Math.round(open * 100))}%, var(--surface))`,
      }}
    />
  );
}

function stroke(alarm: "warning" | "critical" | undefined): string {
  return alarm ? ` in-alarm-${alarm}` : "";
}

export function Mimic({ plant, plc }: { plant: PlantState; plc: PlcStatus | null }) {
  const { boiler, turbine, actuators, condenser, emissions } = plant;
  const alarms = equipmentInAlarm(plc?.active_conditions ?? []);
  const levelFraction = Math.min(Math.max(boiler.water_level_m / DRUM_LEVEL_SPAN_M, 0), 1);
  const waterHeight = DRUM.height * levelFraction;
  const firing = Math.min(Math.max(actuators.fuel_valve_position, 0), 1);
  const burning = boiler.fuel_flow_kg_s > 0.05;
  const estop = plc?.emergency_stop_active ?? false;

  return (
    <svg
      className="mimic"
      viewBox="0 0 1000 560"
      role="img"
      aria-label="Process mimic of the boiler and turbine"
    >
      <defs>
        <clipPath id="drum-clip">
          <rect x={DRUM.x} y={DRUM.y} width={DRUM.width} height={DRUM.height} rx={30} />
        </clipPath>
      </defs>

      {/* Furnace and convective pass */}
      <rect
        x={80}
        y={150}
        width={200}
        height={320}
        rx={6}
        className={`equipment${stroke(alarms.furnace)}`}
      />
      <text x={180} y={175} textAnchor="middle" className="label">
        Furnace
      </text>
      <Reading
        x={180}
        y={198}
        anchor="middle"
        label="Exit gas"
        value={formatReading(kelvinToCelsius(boiler.flue_gas_temp_k), 0)}
        unit="°C"
        quality={qualityOf(plant, "furnace_gas_temp")}
        testId="mimic-furnace-gas"
      />
      {burning && (
        <ellipse
          cx={180}
          cy={440}
          rx={25 + 60 * firing}
          ry={8 + 22 * firing}
          className="flame"
          opacity={0.35 + 0.6 * firing}
        />
      )}
      <Reading
        x={180}
        y={410 - 22 * firing}
        anchor="middle"
        label="Heat"
        value={formatReading(wattsToMegawatts(boiler.heat_release_w), 0)}
        unit="MW"
      />
      <rect x={290} y={150} width={70} height={320} rx={6} className="equipment" />

      {/* Fuel */}
      <path d="M 10 440 L 80 440" className="pipe fuel" />
      <Valve x={45} y={440} position={actuators.fuel_valve_position} />
      <Reading
        x={84}
        y={500}
        label="Fuel"
        value={formatReading(kilogramsPerSecondToTonnesPerHour(boiler.fuel_flow_kg_s), 1)}
        unit={`t/h · valve ${formatReading(fractionToPercent(actuators.fuel_valve_position), 1)} %`}
        quality={qualityOf(plant, "fuel_flow")}
        testId="mimic-fuel"
      />

      {/* Drum with its level */}
      <rect
        x={DRUM.x}
        y={DRUM.y}
        width={DRUM.width}
        height={DRUM.height}
        rx={30}
        className={`equipment${stroke(alarms.drum)}`}
      />
      <rect
        x={DRUM.x}
        y={DRUM.y + DRUM.height - waterHeight}
        width={DRUM.width}
        height={waterHeight}
        className="drum-water"
        clipPath="url(#drum-clip)"
      />
      <line x1={110} y1={110} x2={110} y2={150} className="pipe water" />
      <line x1={250} y1={110} x2={250} y2={150} className="pipe water" />
      <Reading
        x={DRUM.x}
        y={36}
        label="Drum"
        value={formatReading(pascalsToBar(boiler.pressure_pa), 1)}
        unit="bar"
        quality={qualityOf(plant, "drum_pressure")}
        testId="mimic-drum-pressure"
      />
      <Reading
        x={DRUM.x + 150}
        y={36}
        label="Level"
        value={formatReading(boiler.water_level_m, 2)}
        unit="m"
        quality={qualityOf(plant, "drum_level")}
        testId="mimic-drum-level"
      />

      {/* Superheater and main steam */}
      <path
        d="M 320 80 L 320 170 L 300 185 L 350 200 L 300 215 L 350 230 L 300 245 L 355 255 L 360 255"
        className={`pipe steam${stroke(alarms.superheater)}`}
      />
      <path d="M 360 255 L 560 255" className="pipe steam" />
      <Reading
        x={372}
        y={236}
        label="Main steam"
        value={formatReading(kelvinToCelsius(turbine.steam_temp_in_k), 1)}
        unit="°C"
        quality={qualityOf(plant, "steam_temp")}
        testId="mimic-steam-temperature"
      />
      <Reading
        x={372}
        y={218}
        label="Flow"
        value={formatReading(kilogramsPerSecondToTonnesPerHour(turbine.steam_flow_kg_s), 0)}
        unit="t/h"
        quality={qualityOf(plant, "steam_flow")}
        testId="mimic-steam-flow"
      />
      <path d="M 410 290 L 410 255" className="pipe water" />
      <Valve x={410} y={285} position={actuators.spray_valve_position} vertical />
      <Reading
        x={424}
        y={290}
        label="Spray"
        value={formatReading(fractionToPercent(actuators.spray_valve_position), 1)}
        unit="%"
      />
      <Valve x={520} y={255} position={actuators.steam_valve_position} />
      <Reading
        x={520}
        y={200}
        anchor="middle"
        label="Valve"
        value={formatReading(fractionToPercent(actuators.steam_valve_position), 1)}
        unit="%"
        testId="mimic-turbine-valve"
      />

      {/* Turbine and generator */}
      <polygon points="560,225 700,190 700,320 560,285" className="equipment" />
      <text x={630} y={260} textAnchor="middle" className="label">
        Turbine
      </text>
      <line x1={700} y1={255} x2={780} y2={255} className="pipe gas" />
      <circle cx={830} cy={255} r={50} className={`equipment${stroke(alarms.generator)}`} />
      <text x={830} y={262} textAnchor="middle" className="label">
        G
      </text>
      <Reading
        x={830}
        y={335}
        anchor="middle"
        label=""
        value={formatReading(wattsToMegawatts(turbine.electrical_power_w), 1)}
        unit="MW"
        quality={qualityOf(plant, "electrical_power")}
        testId="mimic-power"
      />

      {/* Condenser */}
      <path d="M 630 303 L 630 400" className="pipe steam" />
      <rect x={560} y={400} width={160} height={70} rx={6} className="equipment" />
      <text x={640} y={425} textAnchor="middle" className="label">
        Condenser
      </text>
      <Reading
        x={640}
        y={450}
        anchor="middle"
        label=""
        value={formatReading(pascalsToKilopascals(condenser.backpressure_pa), 1)}
        unit="kPa"
      />

      {/* Feedwater */}
      <path
        d="M 600 470 L 600 500 L 340 500 L 340 455 L 300 440 L 350 425 L 300 410 L 350 395 L 300 380 L 285 370 L 285 110"
        className={`pipe water${stroke(alarms.feedwater)}`}
      />
      <circle cx={520} cy={500} r={14} className="equipment" />
      <text x={520} y={529} textAnchor="middle" className="label">
        Pump
      </text>
      <Valve x={440} y={500} position={actuators.feedwater_valve_position} />
      <Reading
        x={620}
        y={525}
        label="Feedwater"
        value={formatReading(kilogramsPerSecondToTonnesPerHour(boiler.feedwater_flow_kg_s), 0)}
        unit={`t/h · valve ${formatReading(fractionToPercent(actuators.feedwater_valve_position), 1)} %`}
        quality={qualityOf(plant, "feedwater_flow")}
        testId="mimic-feedwater"
      />

      {/* Flue gas to the stack */}
      <path d="M 360 462 L 395 462" className="pipe gas" />
      <rect x={395} y={330} width={30} height={140} className="equipment" />
      <Reading
        x={434}
        y={345}
        label="Stack"
        value={formatReading(kelvinToCelsius(boiler.stack_temp_k), 0)}
        unit="°C"
      />
      <Reading
        x={434}
        y={365}
        label="NOx"
        value={formatReading(emissions.nox_ppmv, 1)}
        unit="ppmv"
      />
      <Reading
        x={434}
        y={385}
        label="CO2"
        value={formatReading(emissions.co2_intensity_kg_per_mwh, 0)}
        unit="kg/MWh"
      />

      {estop && (
        <text x={990} y={30} textAnchor="end" className="value" style={{ fill: "var(--crit)" }}>
          E-STOP — fuel shut off
        </text>
      )}
    </svg>
  );
}
