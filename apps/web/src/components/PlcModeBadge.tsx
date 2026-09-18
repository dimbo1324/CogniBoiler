import type { PlcStatus } from "../api/types";

export function PlcModeBadge({ plc }: { plc: PlcStatus | null }) {
  if (plc === null) {
    return <span className="badge">PLC —</span>;
  }
  if (plc.emergency_stop_active) {
    return (
      <span className="badge crit" aria-label="PLC mode">
        E-STOP
      </span>
    );
  }
  return (
    <span className={`badge ${plc.mode === "auto" ? "ok" : "warn"}`} aria-label="PLC mode">
      {plc.mode.toUpperCase()}
    </span>
  );
}
