import type { PlcStatus } from "../api/types";
import { formatQuantity, formatReading, parameterLabel, wattsToMegawatts } from "../units";
import { PlcModeBadge } from "./PlcModeBadge";

export function tripDescription(plc: PlcStatus): string | null {
  if (!plc.trip_cause) {
    return plc.emergency_stop_active ? "Manual trip" : null;
  }
  const cause = plc.trip_cause;
  const unit = cause.parameter.endsWith("_pa") ? "Pa" : cause.parameter.endsWith("_k") ? "K" : "m";
  return `${parameterLabel(cause.parameter)} ${formatQuantity(cause.value, unit)} (limit ${formatQuantity(cause.threshold, unit)})`;
}

export function PlcPanel({ plc }: { plc: PlcStatus | null }) {
  if (plc === null) {
    return (
      <section className="panel" aria-label="PLC">
        <h2>PLC</h2>
        <p className="muted">Waiting for the PLC…</p>
      </section>
    );
  }
  const trip = tripDescription(plc);
  return (
    <section className="panel" aria-label="PLC">
      <h2>
        PLC <PlcModeBadge plc={plc} />
      </h2>
      {plc.emergency_stop_active && (
        <p className="error" role="status">
          Tripped: {trip}
          {plc.reset_blockers.length > 0 && (
            <>
              <br />
              Reset blocked: {plc.reset_blockers.join("; ")}
            </>
          )}
          {plc.reset_permitted && (
            <>
              <br />
              The cause has cleared; an engineer may reset.
            </>
          )}
        </p>
      )}
      <dl className="kv">
        <dt>Load demand</dt>
        <dd data-testid="plc-load-demand">
          {formatReading(wattsToMegawatts(plc.load_demand_w), 1)} MW
        </dd>
        <dt>Load setpoint (ramped)</dt>
        <dd>{formatReading(wattsToMegawatts(plc.load_setpoint_w), 1)} MW</dd>
        <dt>Pressure setpoint</dt>
        <dd>{formatQuantity(plc.active_setpoints.pressure_pa, "Pa")}</dd>
        <dt>Level setpoint</dt>
        <dd>{formatQuantity(plc.active_setpoints.water_level_m, "m")}</dd>
        <dt>Steam temperature setpoint</dt>
        <dd>{formatQuantity(plc.active_setpoints.steam_temp_k, "K")}</dd>
        <dt>Warnings / trips</dt>
        <dd>
          {plc.warning_count} / {plc.trip_count}
        </dd>
      </dl>
    </section>
  );
}
