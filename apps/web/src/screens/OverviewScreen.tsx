import { Link } from "react-router";

import { isUnacknowledged, sortAlarms, useActiveAlarms } from "../alarms/queries";
import type { PlantState, PlcEvent } from "../api/types";
import { Mimic } from "../components/Mimic";
import { PlcPanel } from "../components/PlcPanel";
import { useLive } from "../live/LiveProvider";
import {
  formatDateTime,
  formatDuration,
  formatQuantity,
  formatReading,
  fractionToPercent,
  parameterLabel,
} from "../units";

function SimulationPanel({ plant }: { plant: PlantState }) {
  const { simulation, faults, performance, sensors } = plant;
  const degraded = sensors.filter((sensor) => sensor.quality !== "good");
  return (
    <section className="panel" aria-label="Unit">
      <h2>Unit</h2>
      <dl className="kv">
        <dt>Scenario</dt>
        <dd data-testid="scenario">{simulation.scenario}</dd>
        <dt>Simulated time</dt>
        <dd>
          {formatDuration(simulation.simulation_time_s)}
          {simulation.run_state === "paused"
            ? " (paused)"
            : ` at ${formatReading(simulation.speed_factor, 0)}×`}
        </dd>
        <dt>Net efficiency</dt>
        <dd>{formatReading(fractionToPercent(performance.net_efficiency), 2)} %</dd>
        <dt>Boiler efficiency</dt>
        <dd>{formatReading(fractionToPercent(performance.boiler_efficiency), 2)} %</dd>
        <dt>Equipment health</dt>
        <dd>{formatReading(plant.health.overall_health_pct, 1)} %</dd>
        <dt>Active faults</dt>
        <dd data-testid="active-faults">
          {faults.length === 0 ? "none" : faults.map((fault) => fault.label).join(", ")}
        </dd>
        {degraded.length > 0 && (
          <>
            <dt>Instruments</dt>
            <dd>
              {degraded
                .map((sensor) => `${sensor.sensor_id.replaceAll("_", " ")}: ${sensor.quality}`)
                .join(", ")}
            </dd>
          </>
        )}
      </dl>
    </section>
  );
}

function ActiveAlarmsPanel() {
  const { data: alarms = [], isError } = useActiveAlarms();
  const standing = sortAlarms(alarms).slice(0, 6);
  return (
    <section className="panel" aria-label="Active alarms">
      <h2>
        Active alarms <Link to="/alarms">all</Link>
      </h2>
      {isError && <p className="error">Alarms are unavailable.</p>}
      {!isError && standing.length === 0 && <p className="muted">No active alarms.</p>}
      <ul>
        {standing.map((alarm) => (
          <li
            key={alarm.id}
            className={
              alarm.severity === "critical" && isUnacknowledged(alarm) ? "error flashing" : ""
            }
          >
            <strong>{alarm.severity}</strong> {parameterLabel(alarm.parameter)} —{" "}
            {formatQuantity(alarm.value, alarm.unit)}
            {isUnacknowledged(alarm) ? " (unacknowledged)" : ""}
          </li>
        ))}
      </ul>
    </section>
  );
}

function describeEvent(event: PlcEvent): string {
  const detail = Object.entries(event.detail)
    .filter(([, value]) => typeof value !== "object")
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
  return `${event.kind.replaceAll("_", " ")}${event.operator_id ? ` by ${event.operator_id}` : ""}${detail ? ` (${detail})` : ""}`;
}

function EventsPanel({ events }: { events: readonly PlcEvent[] }) {
  return (
    <section className="panel" aria-label="PLC events">
      <h2>PLC events</h2>
      {events.length === 0 ? (
        <p className="muted">No events since this page opened.</p>
      ) : (
        <ul>
          {events.slice(0, 8).map((event) => (
            <li key={event.event_id}>
              <span className="muted">{formatDateTime(event.timestamp_ms)}</span>{" "}
              {describeEvent(event)}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function OverviewScreen() {
  const live = useLive();
  return (
    <div className="grid overview-grid">
      <div className="stack">
        <section className="panel" aria-label="Mimic">
          {live.plant ? (
            <Mimic plant={live.plant} plc={live.plc} />
          ) : (
            <p className="muted" role="status">
              Waiting for the first plant state…
            </p>
          )}
          {live.plantReceivedAtMs !== null && (
            <p className="muted">Updated {formatDateTime(live.plantReceivedAtMs)}</p>
          )}
        </section>
        <EventsPanel events={live.events} />
      </div>
      <div className="stack">
        <PlcPanel plc={live.plc} />
        <ActiveAlarmsPanel />
        {live.plant && <SimulationPanel plant={live.plant} />}
      </div>
    </div>
  );
}
