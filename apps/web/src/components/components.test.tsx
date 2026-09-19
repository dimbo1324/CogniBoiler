import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useActiveAlarms } from "../alarms/queries";
import { ApiError } from "../api/http";
import type { Alarm } from "../api/types";
import { useCan, useRole } from "../session/SessionProvider";
import { alarm, plcStatus } from "../test/fixtures";
import { AlarmBanner } from "./AlarmBanner";
import { CommandResult } from "./CommandResult";
import { Pager } from "./Pager";
import { PlcModeBadge } from "./PlcModeBadge";
import { RequirePermission } from "./RequirePermission";

const horn = vi.hoisted(() => ({ start: vi.fn(), stop: vi.fn() }));

vi.mock("../alarms/horn", () => ({
  Horn: class {
    start = horn.start;
    stop = horn.stop;
  },
}));

vi.mock("../alarms/queries", async (importOriginal) => {
  const original = await importOriginal<typeof import("../alarms/queries")>();
  return { ...original, useActiveAlarms: vi.fn() };
});

vi.mock("../session/SessionProvider", () => ({ useCan: vi.fn(), useRole: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Pager", () => {
  it("shows the range and moves by one page", () => {
    const onChange = vi.fn();
    render(<Pager offset={50} limit={50} total={120} onChange={onChange} />);
    expect(screen.getByText("51–100 of 120")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    expect(onChange.mock.calls).toEqual([[100], [0]]);
  });

  it("disables what leads nowhere", () => {
    render(<Pager offset={0} limit={50} total={0} onChange={vi.fn()} />);
    expect(screen.getByText("0–0 of 0")).toBeDefined();
    expect((screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect((screen.getByRole("button", { name: "Next" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("never goes before the first row", () => {
    const onChange = vi.fn();
    render(<Pager offset={20} limit={50} total={200} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    expect(onChange).toHaveBeenCalledWith(0);
  });
});

describe("PlcModeBadge", () => {
  it("shows the mode, a trip, or nothing known", () => {
    const { rerender } = render(<PlcModeBadge plc={null} />);
    expect(screen.getByText("PLC —")).toBeDefined();
    rerender(<PlcModeBadge plc={plcStatus()} />);
    expect(screen.getByLabelText("PLC mode").textContent).toBe("AUTO");
    expect(screen.getByLabelText("PLC mode").className).toContain("ok");
    rerender(<PlcModeBadge plc={plcStatus({ mode: "manual" })} />);
    expect(screen.getByLabelText("PLC mode").className).toContain("warn");
    rerender(<PlcModeBadge plc={plcStatus({ mode: "estop", emergency_stop_active: true })} />);
    expect(screen.getByLabelText("PLC mode").textContent).toBe("E-STOP");
  });
});

describe("CommandResult", () => {
  it("says nothing before a command", () => {
    const { container } = render(<CommandResult result={undefined} error={null} />);
    expect(container.textContent).toBe("");
  });

  it("reports acceptance and refusal", () => {
    const { rerender } = render(
      <CommandResult result={{ accepted: true, reason: "" }} error={null} />,
    );
    expect(screen.getByRole("status").textContent).toBe("Accepted.");
    rerender(<CommandResult result={{ accepted: false, reason: "E-Stop active" }} error={null} />);
    expect(screen.getByRole("alert").textContent).toBe("Refused: E-Stop active");
  });

  it("explains a gateway error instead", () => {
    const error = new ApiError({
      status: 422,
      code: "request.invalid",
      title: "Unprocessable",
      detail: "The request did not pass validation.",
      errors: [{ location: "body.load_w", message: "too large" }],
      retryAfterS: null,
    });
    render(<CommandResult result={{ accepted: true, reason: "" }} error={error} />);
    expect(screen.getByRole("alert").textContent).toBe(
      "The request did not pass validation. body.load_w: too large",
    );
  });
});

describe("RequirePermission", () => {
  it("shows the screen to a role that may use it", () => {
    vi.mocked(useCan).mockReturnValue(true);
    vi.mocked(useRole).mockReturnValue("admin");
    render(
      <RequirePermission permission="manage_users">
        <p>user list</p>
      </RequirePermission>,
    );
    expect(screen.getByText("user list")).toBeDefined();
  });

  it("says why another role sees nothing", () => {
    vi.mocked(useCan).mockReturnValue(false);
    vi.mocked(useRole).mockReturnValue("viewer");
    render(
      <RequirePermission permission="manage_users">
        <p>user list</p>
      </RequirePermission>,
    );
    expect(screen.getByRole("status").textContent).toBe(
      "This screen is not available to the viewer role.",
    );
    expect(screen.queryByText("user list")).toBeNull();
  });
});

function renderBanner(alarms: Alarm[]) {
  vi.mocked(useActiveAlarms).mockReturnValue({ data: alarms } as ReturnType<
    typeof useActiveAlarms
  >);
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter>
        <AlarmBanner />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AlarmBanner", () => {
  beforeEach(() => {
    horn.start.mockClear();
    horn.stop.mockClear();
  });

  it("stays hidden while every alarm is acknowledged", () => {
    const { container } = renderBanner([alarm({ state: "ACTIVE_ACK", acknowledged: true })]);
    expect(container.textContent).toBe("");
    expect(horn.start).not.toHaveBeenCalled();
  });

  it("counts unacknowledged warnings without sounding", () => {
    renderBanner([alarm(), alarm({ id: 2 })]);
    expect(screen.getByRole("alert").textContent).toContain("2 unacknowledged alarms");
    expect(screen.queryByRole("button", { name: "Silence" })).toBeNull();
    expect(horn.start).not.toHaveBeenCalled();
  });

  it("flashes and sounds for a critical alarm until silenced", () => {
    renderBanner([
      alarm({ id: 5, severity: "critical", parameter: "water_level_m", message: "level low" }),
    ]);
    const banner = screen.getByRole("alert");
    expect(banner.className).toContain("flashing");
    expect(banner.textContent).toContain("1 unacknowledged critical alarm");
    expect(banner.textContent).toContain("level low");
    expect(horn.start).toHaveBeenCalled();
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Silence" }));
    });
    expect(horn.stop).toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Silence" })).toBeNull();
    expect(screen.getByRole("link", { name: "Open alarms" }).getAttribute("href")).toBe("/alarms");
  });
});
