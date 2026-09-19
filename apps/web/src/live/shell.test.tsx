import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useActiveAlarms } from "../alarms/queries";
import { refreshSession, signOut } from "../api/endpoints";
import type { RealtimeOptions } from "../api/realtime";
import type { AlarmChange, DataFrame } from "../api/types";
import { Layout } from "../components/Layout";
import { SessionProvider, useCan, useSession, useUser } from "../session/SessionProvider";
import { SessionManager } from "../session/session";
import { plantState, plcStatus, tokens } from "../test/fixtures";
import { LiveProvider, useLive, useLiveStore } from "./LiveProvider";

const realtime = vi.hoisted(() => ({
  clients: [] as {
    options: RealtimeOptions;
    start: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    renew: ReturnType<typeof vi.fn>;
  }[],
}));

vi.mock("../api/realtime", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/realtime")>();
  return {
    ...original,
    RealtimeClient: class {
      readonly start = vi.fn();
      readonly stop = vi.fn();
      readonly renew = vi.fn();
      constructor(readonly options: RealtimeOptions) {
        realtime.clients.push(this);
      }
    },
  };
});

vi.mock("../api/endpoints", () => ({
  refreshSession: vi.fn(),
  signIn: vi.fn(),
  signOut: vi.fn(),
}));

vi.mock("../alarms/queries", async (importOriginal) => {
  const original = await importOriginal<typeof import("../alarms/queries")>();
  return { ...original, useActiveAlarms: vi.fn() };
});

function Probe() {
  const live = useLive();
  const { state } = useSession();
  return (
    <p data-testid="probe">
      {state.status} {live.connection} {live.plant === null ? "no-plant" : "plant"}{" "}
      {live.plc?.mode ?? "no-plc"} {live.events.length} events
    </p>
  );
}

function client() {
  const current = realtime.clients.at(-1);
  if (current === undefined) {
    throw new Error("no realtime client was created");
  }
  return current;
}

function renderShell(node: ReactNode, queryClient = new QueryClient()) {
  const manager = new SessionManager();
  const view = render(
    <QueryClientProvider client={queryClient}>
      <SessionProvider manager={manager}>
        <LiveProvider>
          <MemoryRouter>{node}</MemoryRouter>
        </LiveProvider>
      </SessionProvider>
    </QueryClientProvider>,
  );
  return { manager, view };
}

function SignedIn({ children }: { children: ReactNode }) {
  const { state } = useSession();
  return state.status === "signed_in" ? children : <p>{state.status}</p>;
}

const alarmChange = { alarm: { id: 1 }, transition: { id: 1 } } as unknown as AlarmChange;

beforeEach(() => {
  realtime.clients.length = 0;
  vi.mocked(refreshSession).mockResolvedValue(
    tokens({ access_expires_at_ms: Date.now() + 900_000 }),
  );
  vi.mocked(signOut).mockResolvedValue(undefined as never);
  vi.mocked(useActiveAlarms).mockReturnValue({ data: [] } as unknown as ReturnType<
    typeof useActiveAlarms
  >);
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  document.documentElement.removeAttribute("data-theme");
});

describe("LiveProvider", () => {
  it("opens the three live channels with the session's token", async () => {
    const { manager } = renderShell(<Probe />);
    await waitFor(() => {
      expect(manager.accessToken()).toBe("access-1");
    });
    const { options, start } = client();
    expect(start).toHaveBeenCalledTimes(1);
    expect(options.channels).toEqual(["telemetry", "plc", "alarms"]);
    expect(options.maxRateHz).toBe(2);
    expect(options.url).toMatch(/^ws:\/\/.+\/ws$/u);
    expect(options.token()).toBe("access-1");
  });

  it("applies frames to the store and reloads what they invalidate", async () => {
    const queryClient = new QueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    renderShell(<Probe />, queryClient);
    await screen.findByText(/signed_in/u);
    const frames: DataFrame[] = [
      { type: "data", channel: "telemetry", kind: "state", ts_ms: 1, data: plantState() },
      {
        type: "data",
        channel: "plc",
        kind: "status",
        ts_ms: 1,
        data: plcStatus({ mode: "manual" }),
      },
      {
        type: "data",
        channel: "plc",
        kind: "event",
        ts_ms: 1,
        data: {
          event_id: "e1",
          kind: "mode_changed",
          source_service: "plc-controller",
          operator_id: "operator",
          detail: {},
          timestamp_ms: 1,
        },
      },
      { type: "data", channel: "alarms", kind: "change", ts_ms: 1, data: alarmChange },
    ];
    act(() => {
      client().options.handlers.state("live");
      for (const frame of frames) {
        client().options.handlers.frame(frame);
      }
    });
    expect(screen.getByTestId("probe").textContent).toBe("signed_in live plant manual 1 events");
    const keys = invalidate.mock.calls.map((call) => call[0]?.queryKey);
    expect(keys).toEqual([["plc"], ["alarms"]]);
    act(() => {
      client().options.handlers.resync();
    });
    expect(invalidate).toHaveBeenLastCalledWith();
  });

  it("renews a refused token through the session and hands every new token over", async () => {
    const { manager } = renderShell(<Probe />);
    await screen.findByText(/signed_in/u);
    vi.mocked(refreshSession).mockResolvedValue(
      tokens({ access_token: "access-2", access_expires_at_ms: Date.now() + 900_000 }),
    );
    await act(async () => {
      expect(await client().options.handlers.unauthorized()).toBe("access-2");
    });
    expect(client().renew).toHaveBeenCalledWith("access-2");
    expect(manager.accessToken()).toBe("access-2");
  });

  it("closes the channels when the console unmounts", async () => {
    const { view } = renderShell(<Probe />);
    await screen.findByText(/signed_in/u);
    const opened = client();
    view.unmount();
    expect(opened.stop).toHaveBeenCalledTimes(1);
  });

  it("refuses to be used outside its provider", () => {
    function Orphan() {
      useLiveStore();
      return null;
    }
    const quiet = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(() => render(<Orphan />)).toThrow("useLive must be used inside LiveProvider");
    quiet.mockRestore();
  });
});

describe("SessionProvider", () => {
  it("clears cached data when the session ends", async () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["alarms", "active"], ["cached"]);
    const { manager } = renderShell(<Probe />, queryClient);
    await screen.findByText(/signed_in/u);
    expect(queryClient.getQueryData(["alarms", "active"])).toEqual(["cached"]);
    await act(async () => {
      await manager.signOut();
    });
    expect(screen.getByTestId("probe").textContent).toContain("signed_out");
    expect(queryClient.getQueryData(["alarms", "active"])).toBeUndefined();
  });

  it("answers permissions by the signed-in role", async () => {
    function Rights() {
      const { state } = useSession();
      const setLoad = useCan("set_load");
      const manageUsers = useCan("manage_users");
      return (
        <p data-testid="rights">
          {state.status} {String(setLoad)} {String(manageUsers)}
        </p>
      );
    }
    renderShell(<Rights />);
    expect(screen.getByTestId("rights").textContent).toBe("restoring false false");
    await waitFor(() => {
      expect(screen.getByTestId("rights").textContent).toBe("signed_in true false");
    });
  });

  it("refuses the user of a session that is not signed in, and use outside it", () => {
    function NeedsUser() {
      useUser();
      return null;
    }
    const quiet = vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.mocked(refreshSession).mockReturnValue(new Promise(() => undefined));
    expect(() => render(<NeedsUser />)).toThrow("useSession must be used inside SessionProvider");
    expect(() =>
      render(
        <QueryClientProvider client={new QueryClient()}>
          <SessionProvider manager={new SessionManager()}>
            <NeedsUser />
          </SessionProvider>
        </QueryClientProvider>,
      ),
    ).toThrow("useUser needs a signed-in session");
    quiet.mockRestore();
  });
});

describe("Layout", () => {
  it("shows the screens the role may open, the connection and the user", async () => {
    renderShell(
      <SignedIn>
        <Layout />
      </SignedIn>,
    );
    const nav = await screen.findByRole("navigation", { name: "Screens" });
    expect(Array.from(nav.querySelectorAll("a")).map((link) => link.textContent)).toEqual([
      "Overview",
      "Trends",
      "Alarms",
      "Control",
      "Platform",
    ]);
    expect(screen.getByLabelText("Signed in as").textContent).toBe("operator (operator)");
    expect(screen.getByLabelText("Live data connection").textContent).toBe("Connecting…");
    act(() => {
      client().options.handlers.state("live");
    });
    expect(screen.getByLabelText("Live data connection").className).toBe("badge ok");
    act(() => {
      client().options.handlers.state("reconnecting");
    });
    expect(screen.getByLabelText("Live data connection").textContent).toBe("Reconnecting…");
  });

  it("gives an administrator every screen", async () => {
    vi.mocked(refreshSession).mockResolvedValue(
      tokens({ username: "admin", role: "admin", access_expires_at_ms: Date.now() + 900_000 }),
    );
    renderShell(
      <SignedIn>
        <Layout />
      </SignedIn>,
    );
    const nav = await screen.findByRole("navigation", { name: "Screens" });
    expect(nav.querySelectorAll("a")).toHaveLength(8);
  });

  it("cycles the theme and remembers it", async () => {
    renderShell(
      <SignedIn>
        <Layout />
      </SignedIn>,
    );
    const button = await screen.findByRole("button", { name: "Theme: system" });
    await userEvent.click(button);
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(window.localStorage.getItem("cogniboiler.theme")).toBe("dark");
    await userEvent.click(screen.getByRole("button", { name: "Theme: dark" }));
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    await userEvent.click(screen.getByRole("button", { name: "Theme: light" }));
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(window.localStorage.getItem("cogniboiler.theme")).toBeNull();
  });

  it("signs out through the gateway", async () => {
    const { manager } = renderShell(
      <SignedIn>
        <Layout />
      </SignedIn>,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(vi.mocked(signOut)).toHaveBeenCalledTimes(1);
    expect(manager.snapshot().status).toBe("signed_out");
  });
});
