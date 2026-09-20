import type { ReactNode } from "react";

import { Icon, type IconGlyph } from "./Icon";

export type BadgeTone = "neutral" | "ok" | "warn" | "crit";

const TONE_ICON_TONE = {
  neutral: "muted",
  ok: "ok",
  warn: "warn",
  crit: "default",
} as const;

/**
 * A short, coloured state: the live connection, the PLC mode, an alarm severity. The text
 * says what it is; the colour and the glyph only make it faster to find.
 */
export function Badge({
  tone = "neutral",
  glyph,
  label,
  spin = false,
  children,
}: {
  tone?: BadgeTone;
  glyph?: IconGlyph;
  label?: string;
  spin?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={tone === "neutral" ? "badge" : `badge ${tone}`} aria-label={label}>
      {glyph && <Icon glyph={glyph} tone={TONE_ICON_TONE[tone]} spin={spin} />}
      {children}
    </span>
  );
}
