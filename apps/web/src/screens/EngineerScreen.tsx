import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  clearAllFaults,
  clearFault,
  fetchScenarioRuns,
  fetchScenarios,
  injectFault,
  loadScenario,
  pauseSimulation,
  resumeSimulation,
  setSimulationSpeed,
  stepSimulation,
} from "../api/endpoints";
import type { FaultAck, FaultKind, SimulationAck, SimulationStatus } from "../api/types";
import {
  LastActionPanel,
  useConfirmedAction,
  type PendingAction,
} from "../components/ConfirmedAction";
import { Pager } from "../components/Pager";
import { Icon } from "../components/ui/Icon";
import { EmptyNote, ErrorOf } from "../components/ui/Note";
import {
  AuditIcon,
  EngineerIcon,
  PauseIcon,
  RunIcon,
  UnitIcon,
  WarningIcon,
} from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";
import { useLive } from "../live/LiveProvider";
import { formatDateTime, formatDuration, formatReading, parseDecimal } from "../units";
import { queryKeys } from "../api/queryKeys";

type Answer = SimulationAck | FaultAck;

// The gateway's bounds for these two fields (schemas/plant.py): one hour of steps, and a
// fault that takes at most an hour to develop.
const STEPS_MIN = 1;
const STEPS_MAX = 3600;
const FAULT_RAMP_MAX_S = 3600;

interface FaultKindSpec {
  kind: FaultKind;
  label: string;
  target: "none" | "valve" | "sensor";
  /** Admissible severity and its meaning; null when the kind does not use it. */
  severity: { min: number; max: number; initial: number; meaning: string } | null;
}

// Mirrors physics_engine/faults.py: what each fault needs and what its severity means.
export const FAULT_KINDS: readonly FaultKindSpec[] = [
  {
    kind: "feedwater_pump_failure",
    label: "Feedwater pump failure",
    target: "none",
    severity: { min: 0, max: 1, initial: 1, meaning: "fraction of pump capacity lost (1 = trip)" },
  },
  {
    kind: "burner_fouling",
    label: "Burner fouling",
    target: "none",
    severity: { min: 0, max: 0.5, initial: 0.2, meaning: "fraction of combustion efficiency lost" },
  },
  {
    kind: "steam_leak",
    label: "Steam leak",
    target: "none",
    severity: { min: 0, max: 0.3, initial: 0.1, meaning: "leak as a fraction of rated steam flow" },
  },
  { kind: "valve_stuck", label: "Valve stuck", target: "valve", severity: null },
  {
    kind: "sensor_drift",
    label: "Sensor drift",
    target: "sensor",
    severity: {
      min: -0.2,
      max: 0.2,
      initial: 0.05,
      meaning: "drift per minute as a fraction of span",
    },
  },
  { kind: "sensor_failure", label: "Sensor failure", target: "sensor", severity: null },
];

const VALVE_TARGETS = ["fuel", "feedwater", "steam", "spray"] as const;
const SPEEDS = [1, 2, 5, 10, 20, 50] as const;
const RUNS_PAGE = 20;

function faultSpec(kind: FaultKind): FaultKindSpec {
  return FAULT_KINDS.find((spec) => spec.kind === kind) ?? (FAULT_KINDS[0] as FaultKindSpec);
}

function SimulationSection({
  status,
  run,
}: {
  status: SimulationStatus;
  run: (label: string, action: () => Promise<Answer>) => void;
}) {
  const [speed, setSpeed] = useState(String(status.speed_factor));
  const [steps, setSteps] = useState("10");
  const paused = status.run_state === "paused";
  const stepCount = parseDecimal(steps);
  const stepsValid = stepCount !== null && stepCount >= STEPS_MIN && stepCount <= STEPS_MAX;
  return (
    <Panel title="Simulation" glyph={EngineerIcon}>
      <dl className="kv">
        <dt>Scenario</dt>
        <dd>{status.scenario}</dd>
        <dt>State</dt>
        <dd data-testid="simulation-state">
          {status.run_state} at {formatReading(status.speed_factor, 0)}×
        </dd>
        <dt>Simulated time</dt>
        <dd>
          {formatDuration(status.simulation_time_s)} ({status.step_count} steps of {status.step_s}{" "}
          s)
        </dd>
      </dl>
      <div className="row">
        <button
          type="button"
          onClick={() => {
            run(paused ? "Resume" : "Pause", paused ? resumeSimulation : pauseSimulation);
          }}
        >
          <Icon glyph={paused ? RunIcon : PauseIcon} />
          {paused ? "Resume" : "Pause"}
        </button>
        <label>
          Speed{" "}
          <select
            value={speed}
            onChange={(event) => {
              setSpeed(event.target.value);
            }}
          >
            {SPEEDS.map((value) => (
              <option key={value} value={String(value)}>
                {value}×
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => {
            run(`Speed ${speed}×`, () => setSimulationSpeed(Number(speed)));
          }}
        >
          Set speed
        </button>
        <label>
          Steps{" "}
          <input
            type="number"
            min={STEPS_MIN}
            max={STEPS_MAX}
            value={steps}
            onChange={(event) => {
              setSteps(event.target.value);
            }}
          />
        </label>
        <button
          type="button"
          disabled={!paused || !stepsValid}
          onClick={() => {
            if (stepCount === null) {
              return;
            }
            run(`Step ${steps}`, () => stepSimulation(stepCount));
          }}
        >
          Step
        </button>
      </div>
    </Panel>
  );
}

function ScenarioSection({ ask }: { ask: (action: PendingAction<Answer>) => void }) {
  const scenarios = useQuery({
    queryKey: queryKeys.simulation.scenarios,
    queryFn: ({ signal }) => fetchScenarios(signal),
  });
  return (
    <Panel title="Scenarios" glyph={UnitIcon}>
      {scenarios.isError && <ErrorOf error={scenarios.error} />}
      <table>
        <tbody>
          {(scenarios.data?.scenarios ?? []).map((scenario) => (
            <tr key={scenario.name}>
              <td>
                <strong>{scenario.title}</strong>
                {scenario.name === scenarios.data?.current && (
                  <span className="badge">current</span>
                )}
                <div className="muted">{scenario.description}</div>
              </td>
              <td>
                <button
                  type="button"
                  onClick={() => {
                    ask({
                      title: `Load “${scenario.title}”`,
                      body: (
                        <p>
                          The plant restarts from the scenario's initial state and simulated time
                          starts again. Active faults are cleared.
                        </p>
                      ),
                      confirmLabel: "Load scenario",
                      danger: true,
                      run: () => loadScenario(scenario.name),
                    });
                  }}
                >
                  Load…
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function FaultSection({ ask }: { ask: (action: PendingAction<Answer>) => void }) {
  const live = useLive();
  const [kind, setKind] = useState<FaultKind>("feedwater_pump_failure");
  const [target, setTarget] = useState("");
  const [severity, setSeverity] = useState("1");
  const [ramp, setRamp] = useState("0");
  const spec = faultSpec(kind);
  const sensors = live.plant?.sensors.map((sensor) => sensor.sensor_id) ?? [];
  const targets =
    spec.target === "valve" ? [...VALVE_TARGETS] : spec.target === "sensor" ? sensors : [];
  const severityValue = parseDecimal(severity);
  const rampValue = parseDecimal(ramp);
  // A kind without a severity is sent with the contract's default of 1.
  const severityToSend =
    spec.severity === null
      ? 1
      : severityValue !== null &&
          severityValue >= spec.severity.min &&
          severityValue <= spec.severity.max
        ? severityValue
        : null;
  const rampToSend =
    rampValue !== null && rampValue >= 0 && rampValue <= FAULT_RAMP_MAX_S ? rampValue : null;
  const targetValid = spec.target === "none" || targets.includes(target);
  const valid = targetValid && severityToSend !== null && rampToSend !== null;
  const faults = live.plant?.faults ?? [];

  return (
    <Panel title="Faults" glyph={WarningIcon}>
      <div className="form-grid">
        <label>
          Fault
          <select
            value={kind}
            onChange={(event) => {
              const next = faultSpec(event.target.value as FaultKind);
              setKind(next.kind);
              setTarget("");
              setSeverity(String(next.severity?.initial ?? 1));
            }}
          >
            {FAULT_KINDS.map((item) => (
              <option key={item.kind} value={item.kind}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        {spec.target !== "none" && (
          <label>
            {spec.target === "valve" ? "Valve" : "Sensor"}
            <select
              value={target}
              onChange={(event) => {
                setTarget(event.target.value);
              }}
            >
              <option value="">choose…</option>
              {targets.map((item) => (
                <option key={item} value={item}>
                  {item.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
        )}
        {spec.severity !== null && (
          <label>
            Severity ({spec.severity.min} … {spec.severity.max})
            <input
              type="number"
              step={0.05}
              min={spec.severity.min}
              max={spec.severity.max}
              value={severity}
              onChange={(event) => {
                setSeverity(event.target.value);
              }}
            />
            <span className="muted">{spec.severity.meaning}</span>
          </label>
        )}
        <label>
          Develops over [s]
          <input
            type="number"
            min={0}
            max={FAULT_RAMP_MAX_S}
            value={ramp}
            onChange={(event) => {
              setRamp(event.target.value);
            }}
          />
        </label>
        <button
          type="button"
          className="danger"
          disabled={!valid}
          onClick={() => {
            if (!targetValid || severityToSend === null || rampToSend === null) {
              return;
            }
            ask({
              title: `Inject: ${spec.label}`,
              body: (
                <p>
                  {spec.label}
                  {target ? ` on ${target.replaceAll("_", " ")}` : ""}
                  {spec.severity ? `, severity ${severity}` : ""}, developing over {ramp} s of
                  simulated time. The PLC and the alarms react as they would on the real unit.
                </p>
              ),
              confirmLabel: "Inject fault",
              danger: true,
              run: () =>
                injectFault({
                  kind,
                  target,
                  severity: severityToSend,
                  ramp_s: rampToSend,
                }),
            });
          }}
        >
          Inject…
        </button>
      </div>
      <h3>Active faults</h3>
      {faults.length === 0 ? (
        <EmptyNote>None.</EmptyNote>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Fault</th>
              <th>Target</th>
              <th className="number">Severity</th>
              <th className="number">Developed</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {faults.map((fault) => (
              <tr key={fault.fault_id} data-testid={`fault-${fault.kind}`}>
                <td>{fault.label}</td>
                <td>{fault.target || "—"}</td>
                <td className="number">{formatReading(fault.severity, 2)}</td>
                <td className="number">{formatReading(fault.intensity * 100, 0)} %</td>
                <td>
                  <button
                    type="button"
                    onClick={() => {
                      ask({
                        title: `Clear: ${fault.label}`,
                        body: <p>The plant stops deviating; the regulators bring it back.</p>,
                        confirmLabel: "Clear fault",
                        run: () => clearFault(fault.fault_id),
                      });
                    }}
                  >
                    Clear…
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {faults.length > 1 && (
        <button
          type="button"
          onClick={() => {
            ask({
              title: "Clear every fault",
              body: <p>{faults.length} faults are cleared at once.</p>,
              confirmLabel: "Clear all faults",
              run: clearAllFaults,
            });
          }}
        >
          Clear all…
        </button>
      )}
    </Panel>
  );
}

const RUN_KIND: Record<string, string> = {
  scenario: "scenario loaded",
  fault_injected: "fault injected",
  fault_cleared: "fault cleared",
};

function RunsSection() {
  const [offset, setOffset] = useState(0);
  const runs = useQuery({
    queryKey: queryKeys.simulation.runs(offset),
    queryFn: ({ signal }) => fetchScenarioRuns(RUNS_PAGE, offset, signal),
  });
  return (
    <Panel title="Scenario and fault log" glyph={AuditIcon}>
      {runs.isError && <ErrorOf error={runs.error} />}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>At</th>
              <th>By</th>
              <th>What</th>
              <th>Scenario</th>
              <th>Fault</th>
              <th className="number">Simulated time</th>
            </tr>
          </thead>
          <tbody>
            {(runs.data?.items ?? []).map((item) => (
              <tr key={item.id}>
                <td>{formatDateTime(item.at_ms)}</td>
                <td>{item.username}</td>
                <td>{RUN_KIND[item.kind] ?? item.kind}</td>
                <td>{item.scenario}</td>
                <td>{item.fault_label ?? "—"}</td>
                <td className="number">{formatDuration(item.simulation_time_s)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager offset={offset} limit={RUNS_PAGE} total={runs.data?.total ?? 0} onChange={setOffset} />
    </Panel>
  );
}

export function EngineerScreen() {
  const live = useLive();
  const action = useConfirmedAction<Answer>(queryKeys.simulation.all);

  return (
    <div className="stack">
      <LastActionPanel title="Last action" last={action.last} />
      {live.plant ? (
        <SimulationSection status={live.plant.simulation} run={action.runNow} />
      ) : (
        <EmptyNote status>Waiting for the plant…</EmptyNote>
      )}
      <FaultSection ask={action.ask} />
      <ScenarioSection ask={action.ask} />
      <RunsSection />
      {action.dialog}
    </div>
  );
}
