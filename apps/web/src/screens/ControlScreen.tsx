import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import {
  resetEmergencyStop,
  sendValveCommand,
  setControlMode,
  setLoadDemand,
  updateSetpoints,
} from "../api/endpoints";
import type { CommandAck, PlcMode, PlcStatus } from "../api/types";
import { CommandResult } from "../components/CommandResult";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { PlcModeBadge } from "../components/PlcModeBadge";
import { tripDescription } from "../components/PlcPanel";
import { Icon } from "../components/ui/Icon";
import { EmptyNote, ErrorNote } from "../components/ui/Note";
import {
  ControlIcon,
  EmergencyStopIcon,
  OkIcon,
  PlcIcon,
  PowerIcon,
  PressureIcon,
  ResetIcon,
} from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";
import { useLive } from "../live/LiveProvider";
import { useCan } from "../session/SessionProvider";
import {
  barToPascals,
  celsiusToKelvin,
  formatReading,
  fractionToPercent,
  kelvinToCelsius,
  megawattsToWatts,
  pascalsToBar,
  wattsToMegawatts,
} from "../units";

// The rates the PLC ramps at (plc_controller/control.py), for the sentences that promise
// them: an operator asks for a target, never for a step.
const LOAD_RAMP_MW_PER_MIN = 30;
const PRESSURE_RAMP_BAR_PER_MIN = 5;

// The gateway's limits (schemas/command.py), in display units.
export const LIMITS = {
  loadMw: [0, 300],
  pressureBar: [50, 185],
  levelM: [0.5, 9],
  steamTempC: [126.85, 574.85],
  valvePct: [0, 100],
} as const;

interface PendingCommand {
  title: string;
  body: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  run: () => Promise<CommandAck>;
}

export function inRange(value: number, [low, high]: readonly [number, number]): boolean {
  return Number.isFinite(value) && value >= low && value <= high;
}

function NumberField({
  label,
  unit,
  value,
  onChange,
  range,
  step,
}: {
  label: string;
  unit: string;
  value: string;
  onChange: (value: string) => void;
  range: readonly [number, number];
  step: number;
}) {
  const valid = inRange(Number(value), range);
  return (
    <label>
      {label} [{unit}]
      <input
        type="number"
        inputMode="decimal"
        min={range[0]}
        max={range[1]}
        step={step}
        value={value}
        aria-invalid={!valid}
        onChange={(event) => {
          onChange(event.target.value);
        }}
      />
      {!valid && (
        <span className="error">
          {range[0]}–{range[1]} {unit}
        </span>
      )}
    </label>
  );
}

function LoadSection({ plc, ask }: { plc: PlcStatus; ask: (command: PendingCommand) => void }) {
  const [load, setLoad] = useState(() => formatReading(wattsToMegawatts(plc.load_demand_w), 0));
  const target = Number(load);
  return (
    <Panel title="Load" glyph={PowerIcon}>
      <p>
        Demand <strong>{formatReading(wattsToMegawatts(plc.load_demand_w), 1)} MW</strong>, ramped
        setpoint {formatReading(wattsToMegawatts(plc.load_setpoint_w), 1)} MW. The unit follows a
        new demand at {LOAD_RAMP_MW_PER_MIN} MW/min.
      </p>
      <div className="form-grid">
        <NumberField
          label="New demand"
          unit="MW"
          value={load}
          onChange={setLoad}
          range={LIMITS.loadMw}
          step={1}
        />
        <button
          type="button"
          className="primary"
          disabled={!inRange(target, LIMITS.loadMw)}
          onClick={() => {
            ask({
              title: "Change the load demand",
              body: (
                <p>
                  From {formatReading(wattsToMegawatts(plc.load_demand_w), 1)} MW to{" "}
                  <strong>{formatReading(target, 1)} MW</strong>. The unit ramps at{" "}
                  {LOAD_RAMP_MW_PER_MIN} MW/min.
                </p>
              ),
              confirmLabel: `Set ${formatReading(target, 1)} MW`,
              run: () => setLoadDemand(megawattsToWatts(target)),
            });
          }}
        >
          <Icon glyph={PowerIcon} />
          Set load…
        </button>
      </div>
    </Panel>
  );
}

const MODE_TEXT: Record<Exclude<PlcMode, "estop">, string> = {
  auto: "The regulators take over from the current valve positions without a bump.",
  manual: "The PLC holds the current valve positions until an operator moves them.",
};

function ModeSection({ plc, ask }: { plc: PlcStatus; ask: (command: PendingCommand) => void }) {
  const choose = (mode: Exclude<PlcMode, "estop">) => {
    ask({
      title: `Switch the PLC to ${mode.toUpperCase()}`,
      body: <p>{MODE_TEXT[mode]}</p>,
      confirmLabel: `Switch to ${mode.toUpperCase()}`,
      run: () => setControlMode(mode),
    });
  };
  return (
    <Panel title="Mode" glyph={PlcIcon} headline={<PlcModeBadge plc={plc} />}>
      <div className="row">
        <button
          type="button"
          disabled={plc.mode === "auto" || plc.emergency_stop_active}
          onClick={() => {
            choose("auto");
          }}
        >
          AUTO…
        </button>
        <button
          type="button"
          disabled={plc.mode === "manual" || plc.emergency_stop_active}
          onClick={() => {
            choose("manual");
          }}
        >
          MANUAL…
        </button>
        <button
          type="button"
          className="danger"
          disabled={plc.emergency_stop_active}
          onClick={() => {
            ask({
              title: "Trip the unit",
              body: (
                <p>
                  The fuel is shut off at once and the PLC latches ESTOP. Only an engineer can reset
                  the trip, and only once its cause has cleared.
                </p>
              ),
              confirmLabel: "Trip now",
              danger: true,
              run: () => setControlMode("estop"),
            });
          }}
        >
          <Icon glyph={EmergencyStopIcon} />
          Emergency stop…
        </button>
      </div>
    </Panel>
  );
}

const VALVES = [
  ["fuel_valve", "Fuel"],
  ["feedwater_valve", "Feedwater"],
  ["steam_valve", "Turbine"],
  ["spray_valve", "Spray"],
] as const;

type ValveName = (typeof VALVES)[number][0];

function positionsOf(plc: PlcStatus): Record<ValveName, string> {
  const command = plc.latest_command;
  return {
    fuel_valve: formatReading(fractionToPercent(command.fuel_valve), 1),
    feedwater_valve: formatReading(fractionToPercent(command.feedwater_valve), 1),
    steam_valve: formatReading(fractionToPercent(command.steam_valve), 1),
    spray_valve: formatReading(fractionToPercent(command.spray_valve), 1),
  };
}

function ValvesSection({ plc, ask }: { plc: PlcStatus; ask: (command: PendingCommand) => void }) {
  const [positions, setPositions] = useState(() => positionsOf(plc));
  const numbers = VALVES.map(([name]) => Number(positions[name]));
  const valid = numbers.every((value) => inRange(value, LIMITS.valvePct));
  const fraction = (name: ValveName) => Number(positions[name]) / 100;
  return (
    <Panel title="Manual valves" glyph={ControlIcon}>
      <p className="muted">
        Sending positions switches the PLC to MANUAL; the interlocks still trip the unit.
      </p>
      <div className="form-grid">
        {VALVES.map(([name, label]) => (
          <NumberField
            key={name}
            label={label}
            unit="%"
            value={positions[name]}
            onChange={(value) => {
              setPositions({ ...positions, [name]: value });
            }}
            range={LIMITS.valvePct}
            step={0.5}
          />
        ))}
        <button
          type="button"
          onClick={() => {
            setPositions(positionsOf(plc));
          }}
        >
          Current commands
        </button>
        <button
          type="button"
          className="primary"
          disabled={!valid || plc.emergency_stop_active}
          onClick={() => {
            ask({
              title: "Send valve positions",
              body: (
                <p>
                  {VALVES.map(
                    ([name, label]) => `${label} ${formatReading(Number(positions[name]), 1)} %`,
                  ).join(", ")}
                  . The PLC switches to MANUAL and holds them until AUTO is selected.
                </p>
              ),
              confirmLabel: "Send and switch to MANUAL",
              run: () =>
                sendValveCommand({
                  fuel_valve: fraction("fuel_valve"),
                  feedwater_valve: fraction("feedwater_valve"),
                  steam_valve: fraction("steam_valve"),
                  spray_valve: fraction("spray_valve"),
                }),
            });
          }}
        >
          <Icon glyph={ControlIcon} />
          Send positions…
        </button>
      </div>
    </Panel>
  );
}

function SetpointsSection({
  plc,
  ask,
}: {
  plc: PlcStatus;
  ask: (command: PendingCommand) => void;
}) {
  const [pressure, setPressure] = useState(() =>
    formatReading(pascalsToBar(plc.setpoints.pressure_pa), 1),
  );
  const [level, setLevel] = useState(() => formatReading(plc.setpoints.water_level_m, 2));
  const [steam, setSteam] = useState(() =>
    formatReading(kelvinToCelsius(plc.setpoints.steam_temp_k), 1),
  );
  const valid =
    inRange(Number(pressure), LIMITS.pressureBar) &&
    inRange(Number(level), LIMITS.levelM) &&
    inRange(Number(steam), LIMITS.steamTempC);
  return (
    <Panel title="Setpoints" glyph={PressureIcon}>
      <p className="muted">
        The working setpoints ramp to new targets at {PRESSURE_RAMP_BAR_PER_MIN} bar/min.
      </p>
      <div className="form-grid">
        <NumberField
          label="Drum pressure"
          unit="bar"
          value={pressure}
          onChange={setPressure}
          range={LIMITS.pressureBar}
          step={0.5}
        />
        <NumberField
          label="Drum level"
          unit="m"
          value={level}
          onChange={setLevel}
          range={LIMITS.levelM}
          step={0.05}
        />
        <NumberField
          label="Main steam temperature"
          unit="°C"
          value={steam}
          onChange={setSteam}
          range={LIMITS.steamTempC}
          step={0.5}
        />
        <button
          type="button"
          className="primary"
          disabled={!valid}
          onClick={() => {
            ask({
              title: "Change the setpoints",
              body: (
                <p>
                  Drum pressure {pressure} bar, drum level {level} m, main steam {steam} °C.
                </p>
              ),
              confirmLabel: "Apply setpoints",
              run: () =>
                updateSetpoints({
                  pressure_pa: barToPascals(Number(pressure)),
                  water_level_m: Number(level),
                  steam_temp_k: celsiusToKelvin(Number(steam)),
                }),
            });
          }}
        >
          Apply…
        </button>
      </div>
    </Panel>
  );
}

function ResetSection({ plc, ask }: { plc: PlcStatus; ask: (command: PendingCommand) => void }) {
  if (!plc.emergency_stop_active) {
    return null;
  }
  return (
    <Panel title="Emergency stop" label="Emergency stop reset" glyph={EmergencyStopIcon}>
      <ErrorNote glyph={EmergencyStopIcon} status>
        Tripped: {tripDescription(plc)}
      </ErrorNote>
      {plc.reset_permitted ? (
        <p>The cause has cleared; the unit returns to AUTO and ramps back to the load demand.</p>
      ) : (
        <p>Reset is blocked: {plc.reset_blockers.join("; ") || "waiting for the PLC"}.</p>
      )}
      <button
        type="button"
        className="danger"
        onClick={() => {
          ask({
            title: "Reset the emergency stop",
            body: (
              <p>
                The PLC checks the cause again and refuses while it stands. After a reset the unit
                returns to AUTO and ramps to {formatReading(wattsToMegawatts(plc.load_demand_w), 1)}{" "}
                MW.
              </p>
            ),
            confirmLabel: "Reset E-Stop",
            danger: true,
            run: resetEmergencyStop,
          });
        }}
      >
        <Icon glyph={ResetIcon} />
        Reset E-Stop…
      </button>
    </Panel>
  );
}

export function ControlScreen() {
  const live = useLive();
  const queryClient = useQueryClient();
  const [pending, setPending] = useState<PendingCommand | null>(null);
  const [last, setLast] = useState<string | null>(null);
  const mayOperate = useCan("set_load");
  const mayValves = useCan("manual_valves");
  const maySetpoints = useCan("set_setpoints");
  const mayReset = useCan("reset_estop");

  const command = useMutation({
    mutationFn: (run: () => Promise<CommandAck>) => run(),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["plc"] }),
  });

  if (live.plc === null) {
    return <EmptyNote status>Waiting for the PLC…</EmptyNote>;
  }
  const plc = live.plc;
  return (
    <div className="stack">
      {last && (
        <Panel
          title="Last command"
          glyph={OkIcon}
          headline={<span className="muted">: {last}</span>}
        >
          <CommandResult result={command.data} error={command.error} />
        </Panel>
      )}
      {mayOperate && <LoadSection plc={plc} ask={setPending} />}
      {mayOperate && <ModeSection plc={plc} ask={setPending} />}
      {mayReset && <ResetSection plc={plc} ask={setPending} />}
      {mayValves && <ValvesSection plc={plc} ask={setPending} />}
      {maySetpoints && <SetpointsSection plc={plc} ask={setPending} />}
      {pending && (
        <ConfirmDialog
          title={pending.title}
          confirmLabel={pending.confirmLabel}
          danger={pending.danger}
          busy={command.isPending}
          onCancel={() => {
            setPending(null);
          }}
          onConfirm={() => {
            setLast(pending.title);
            command.mutate(pending.run, {
              onSettled: () => {
                setPending(null);
              },
            });
          }}
        >
          {pending.body}
        </ConfirmDialog>
      )}
    </div>
  );
}
