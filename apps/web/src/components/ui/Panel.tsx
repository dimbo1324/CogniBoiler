import type { ReactNode } from "react";

import { Icon, type IconGlyph } from "./Icon";

/**
 * The box every screen is built from: a titled region with optional controls on the right.
 *
 * The title is also the region's accessible name, so a screen reader and a Playwright check
 * find a panel by what its heading says. `label` overrides it when the heading is longer
 * than the name the tests and the operator use.
 */
export function Panel({
  title,
  glyph,
  label,
  headline,
  actions,
  className,
  children,
}: {
  title: string;
  glyph?: IconGlyph;
  label?: string;
  headline?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <section className={className ? `panel ${className}` : "panel"} aria-label={label ?? title}>
      <div className="panel-head">
        <h2>
          {glyph && <Icon glyph={glyph} tone="muted" />}
          {title}
          {headline}
        </h2>
        {actions !== undefined && <div className="panel-actions">{actions}</div>}
      </div>
      {children}
    </section>
  );
}
