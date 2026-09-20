// How a severity looks and reads. The contracts carry it as a plain string, so every screen
// used to compare it to "critical" by hand; this is the one place that knows the word.

import type { IconGlyph, IconTone } from "../components/ui/Icon";
import { CriticalIcon, WarningIcon } from "../components/ui/icons";

export const SEVERITIES = ["warning", "critical"] as const;
export type Severity = (typeof SEVERITIES)[number];

export function isCritical(severity: string): boolean {
  return severity === "critical";
}

/** A severity from a contract. Anything the backend has not classified is a warning. */
export function severityOf(severity: string): Severity {
  return isCritical(severity) ? "critical" : "warning";
}

export function severityGlyph(severity: string): IconGlyph {
  return isCritical(severity) ? CriticalIcon : WarningIcon;
}

export function severityTone(severity: string): IconTone {
  return isCritical(severity) ? "crit" : "warn";
}
