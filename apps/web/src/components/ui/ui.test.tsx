import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { isCritical, severityGlyph, severityOf, severityTone } from "../../alarms/severity";
import { ApiError } from "../../api/http";
import { Badge } from "./Badge";
import { Icon } from "./Icon";
import { EmptyNote, ErrorNote, ErrorOf, InfoNote } from "./Note";
import { CriticalIcon, OkIcon, WarningIcon } from "./icons";
import { Panel } from "./Panel";

afterEach(cleanup);

describe("Icon", () => {
  it("is invisible to assistive technology and carries its tone in a class", () => {
    const { container } = render(<Icon glyph={OkIcon} tone="ok" />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("aria-hidden")).toBe("true");
    expect(svg?.getAttribute("focusable")).toBe("false");
    expect(svg?.getAttribute("class")).toContain("icon-ok");
    expect(svg?.getAttribute("class")).not.toContain("spin");
  });

  it("spins only when asked", () => {
    const { container } = render(<Icon glyph={OkIcon} spin />);
    expect(container.querySelector("svg")?.getAttribute("class")).toContain("spin");
  });
});

describe("Badge", () => {
  it("keeps the class the stylesheet and the tests expect", () => {
    const { rerender } = render(<Badge>PLC —</Badge>);
    expect(screen.getByText("PLC —").className).toBe("badge");
    rerender(
      <Badge tone="ok" glyph={OkIcon} label="Live data connection">
        Live
      </Badge>,
    );
    const badge = screen.getByLabelText("Live data connection");
    expect(badge.className).toBe("badge ok");
    // The glyph adds nothing to what is read out: the word alone is the state.
    expect(badge.textContent).toBe("Live");
  });
});

describe("Panel", () => {
  it("names its region after its title, and takes a name of its own", () => {
    const { rerender } = render(<Panel title="Active alarms">body</Panel>);
    expect(screen.getByRole("region", { name: "Active alarms" })).toBeDefined();
    expect(screen.getByRole("heading", { name: "Active alarms" })).toBeDefined();
    rerender(
      <Panel title="Emergency stop" label="Emergency stop reset">
        body
      </Panel>,
    );
    expect(screen.getByRole("region", { name: "Emergency stop reset" })).toBeDefined();
    expect(screen.getByRole("heading", { name: "Emergency stop" })).toBeDefined();
  });

  it("puts a headline inside the heading and actions beside it", () => {
    render(
      <Panel
        title="PLC"
        glyph={OkIcon}
        headline={<span>AUTO</span>}
        actions={<button type="button">all</button>}
        className="trend-list"
      >
        <p>body</p>
      </Panel>,
    );
    const region = screen.getByRole("region", { name: "PLC" });
    expect(region.className).toBe("panel trend-list");
    expect(within(region).getByRole("heading").textContent).toBe("PLCAUTO");
    expect(within(region).getByRole("button", { name: "all" })).toBeDefined();
  });

  it("renders no action bar when a panel has no actions", () => {
    const { container } = render(<Panel title="Unit">body</Panel>);
    expect(container.querySelector(".panel-actions")).toBeNull();
  });
});

describe("notes", () => {
  it("read exactly as their words, with no blank from the glyph", () => {
    render(<ErrorNote>Alarms are unavailable.</ErrorNote>);
    expect(screen.getByRole("alert").textContent).toBe("Alarms are unavailable.");
  });

  it("can stand as a status instead of interrupting", () => {
    render(
      <ErrorNote glyph={CriticalIcon} status>
        Tripped: drum level
      </ErrorNote>,
    );
    expect(screen.getByRole("status").textContent).toBe("Tripped: drum level");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("announces an empty panel only when asked to", () => {
    const { rerender } = render(<EmptyNote>No active alarms.</EmptyNote>);
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByText("No active alarms.")).toBeDefined();
    rerender(<EmptyNote status>Waiting for the first plant state…</EmptyNote>);
    expect(screen.getByRole("status").textContent).toBe("Waiting for the first plant state…");
  });

  it("frames a remark", () => {
    render(<InfoNote>This screen is not available to the viewer role.</InfoNote>);
    const note = screen.getByRole("status");
    expect(note.className).toBe("note notice");
    expect(note.textContent).toBe("This screen is not available to the viewer role.");
  });

  it("explains a gateway failure through the shared description", () => {
    const error = new ApiError({
      status: 503,
      code: "upstream.unavailable",
      title: "Service unavailable",
      detail: "Historian is unavailable.",
      errors: [],
      retryAfterS: null,
    });
    render(<ErrorOf error={error} />);
    expect(screen.getByRole("alert").textContent).toBe("Historian is unavailable.");
  });
});

describe("severity", () => {
  it("knows one word and treats everything else as a warning", () => {
    expect(isCritical("critical")).toBe(true);
    expect(isCritical("CRITICAL")).toBe(false);
    expect(severityOf("critical")).toBe("critical");
    expect(severityOf("warning")).toBe("warning");
    expect(severityOf("whatever the backend adds next")).toBe("warning");
  });

  it("draws critical and warning differently", () => {
    expect(severityGlyph("critical")).toBe(CriticalIcon);
    expect(severityGlyph("warning")).toBe(WarningIcon);
    expect(severityTone("critical")).toBe("crit");
    expect(severityTone("warning")).toBe("warn");
  });
});
