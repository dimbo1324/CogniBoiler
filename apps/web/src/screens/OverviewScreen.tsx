import { Link } from "react-router";

import { isUnacknowledged, sortAlarms, useActiveAlarms } from "../alarms/queries";
import { isCritical, severityGlyph, severityTone } from "../alarms/severity";
import type { PlantState, PlcEvent } from "../api/types";
import { Mimic } from "../components/Mimic";
import { PlcPanel } from "../components/PlcPanel";
import { Icon } from "../components/ui/Icon";
import { EmptyNote, ErrorNote } from "../components/ui/Note";
import { AlarmsIcon, EventsIcon, OverviewIcon, UnitIcon } from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";
import { RecommendationsPanel } from "../insights/RecommendationsPanel";
import { useLive } from "../live/LiveProvider";
import {
  formatDateTime,
  formatDuration,
  formatQuantity,
  formatReading,
  fractionToPercent,
  parameterLabel,
} from "../units";

/** How many of each list the overview shows before sending the operator to its screen. */
const ALARMS_SHOWN = 6;
const EVENTS_SHOWN = 8;

function SimulationPanel({ plant }: { plant: PlantState }) {
  const { simulation, faults, performance, sensors } = plant;
  const degraded = sensors.filter((sensor) => sensor.quality !== "good");
  return (
    <Panel title="Unit" glyph={UnitIcon}>
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
    </Panel>
  );
}

function ActiveAlarmsPanel() {
  const { data: alarms = [], isError } = useActiveAlarms();
  const standing = sortAlarms(alarms).slice(0, ALARMS_SHOWN);
  return (
    <Panel title="Active alarms" glyph={AlarmsIcon} actions={<Link to="/alarms">all</Link>}>
      {isError && <ErrorNote>Alarms are unavailable.</ErrorNote>}
      {!isError && standing.length === 0 && <EmptyNote>No active alarms.</EmptyNote>}
      <ul className="event-list">
        {standing.map((alarm) => (
          <li
            key={alarm.id}
            className={
              isCritical(alarm.severity) && isUnacknowledged(alarm) ? "error flashing" : ""
            }
          >
            <Icon glyph={severityGlyph(alarm.severity)} tone={severityTone(alarm.severity)} />
            <span>
              <strong>{alarm.severity}</strong> {parameterLabel(alarm.parameter)} —{" "}
              {formatQuantity(alarm.value, alarm.unit)}
              {isUnacknowledged(alarm) ? " (unacknowledged)" : ""}
            </span>
          </li>
        ))}
      </ul>
    </Panel>
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
    <Panel title="PLC events" glyph={EventsIcon}>
      {events.length === 0 ? (
        <EmptyNote>No events since this page opened.</EmptyNote>
      ) : (
        <ul className="event-list">
          {events.slice(0, EVENTS_SHOWN).map((event) => (
            <li key={event.event_id}>
              <span className="muted">{formatDateTime(event.timestamp_ms)}</span>
              <span>{describeEvent(event)}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function OverviewScreen() {
  const live = useLive();
  return (
    <div className="grid overview-grid">
      <div className="stack">
        <Panel title="Mimic" glyph={OverviewIcon}>
          {live.plant ? (
            <Mimic plant={live.plant} plc={live.plc} />
          ) : (
            <EmptyNote status>Waiting for the first plant state…</EmptyNote>
          )}
          {live.plantReceivedAtMs !== null && (
            <p className="muted">Updated {formatDateTime(live.plantReceivedAtMs)}</p>
          )}
        </Panel>
        <EventsPanel events={live.events} />
      </div>
      <div className="stack">
        <PlcPanel plc={live.plc} />
        <ActiveAlarmsPanel />
        <RecommendationsPanel />
        {live.plant && <SimulationPanel plant={live.plant} />}
      </div>
    </div>
  );
}
