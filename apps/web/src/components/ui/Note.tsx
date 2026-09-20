import type { ReactNode } from "react";

import { describeError } from "../../api/http";
import { Icon, type IconGlyph } from "./Icon";
import { CriticalIcon, InfoIcon } from "./icons";

/**
 * The three one-line answers a screen gives when it has no data to show: something failed,
 * nothing is there yet, or here is a fact worth a frame. Every screen used to spell these
 * out itself, and they had drifted apart.
 *
 * The glyph is a sibling of the text, spaced by the stylesheet rather than by a blank of
 * its own, so the line reads exactly as its words — what a test and a screen reader see.
 */

/**
 * A failure the operator must see. `role="alert"` interrupts a screen reader, which is
 * right for something that just happened; `status` is for a condition already standing on
 * the screen, such as a latched trip, which would otherwise interrupt on every re-render.
 */
export function ErrorNote({
  children,
  glyph = CriticalIcon,
  status = false,
}: {
  children: ReactNode;
  glyph?: IconGlyph;
  status?: boolean;
}) {
  return (
    <p className="note error" role={status ? "status" : "alert"}>
      <Icon glyph={glyph} tone="crit" />
      {children}
    </p>
  );
}

/** The same, for an unknown value thrown by the API client. */
export function ErrorOf({ error }: { error: unknown }) {
  return <ErrorNote>{describeError(error)}</ErrorNote>;
}

/**
 * Nothing to show. `status` announces it politely, for a panel whose emptiness is news
 * (a screen waiting for the first plant state), and stays off where it is the normal case.
 */
export function EmptyNote({
  children,
  glyph = InfoIcon,
  status = false,
}: {
  children: ReactNode;
  glyph?: IconGlyph;
  status?: boolean;
}) {
  return (
    <p className="note empty" {...(status ? { role: "status" } : {})}>
      <Icon glyph={glyph} tone="muted" />
      {children}
    </p>
  );
}

/** A framed remark: a rule of the screen, or why a control is missing. */
export function InfoNote({
  children,
  glyph = InfoIcon,
}: {
  children: ReactNode;
  glyph?: IconGlyph;
}) {
  return (
    <p className="note notice" role="status">
      <Icon glyph={glyph} tone="accent" />
      {children}
    </p>
  );
}
