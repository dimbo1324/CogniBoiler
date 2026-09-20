import type { PlcStatus } from "../api/types";
import { Badge } from "./ui/Badge";
import { EmergencyStopIcon, PlcIcon } from "./ui/icons";

/** AUTO, MANUAL or a latched trip — the first thing an operator looks for. */
export function PlcModeBadge({ plc }: { plc: PlcStatus | null }) {
  if (plc === null) {
    return <Badge glyph={PlcIcon}>PLC —</Badge>;
  }
  if (plc.emergency_stop_active) {
    return (
      <Badge tone="crit" glyph={EmergencyStopIcon} label="PLC mode">
        E-STOP
      </Badge>
    );
  }
  return (
    <Badge tone={plc.mode === "auto" ? "ok" : "warn"} glyph={PlcIcon} label="PLC mode">
      {plc.mode.toUpperCase()}
    </Badge>
  );
}
