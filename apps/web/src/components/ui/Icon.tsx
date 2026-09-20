import type { LucideProps } from "lucide-react";
import type { ComponentType } from "react";

export type IconGlyph = ComponentType<LucideProps>;
export type IconTone = "default" | "muted" | "accent" | "ok" | "warn" | "crit";

const TONE_CLASS: Record<IconTone, string> = {
  default: "",
  muted: "icon-muted",
  accent: "icon-accent",
  ok: "icon-ok",
  warn: "icon-warn",
  crit: "icon-crit",
};

/**
 * A glyph beside a label. An icon never carries meaning on its own in this console — a
 * control room is read at a glance and in a hurry — so it is hidden from assistive
 * technology, which reads the text next to it. Size and colour come from the stylesheet.
 */
export function Icon({
  glyph: Glyph,
  tone = "default",
  spin = false,
}: {
  glyph: IconGlyph;
  tone?: IconTone;
  spin?: boolean;
}) {
  const classes = ["icon", TONE_CLASS[tone], spin ? "spin" : ""].filter(Boolean).join(" ");
  return <Glyph className={classes} aria-hidden="true" focusable="false" />;
}
