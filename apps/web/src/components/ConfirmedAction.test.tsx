import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/http";
import type { CommandAck } from "../api/types";
import { LastActionPanel, useConfirmedAction, type PendingAction } from "./ConfirmedAction";

const accepted: CommandAck = { accepted: true, reason: "", timestamp_ms: 1 };

function Harness({
  run,
  onSuccess,
}: {
  run: () => Promise<CommandAck>;
  onSuccess?: (action: PendingAction<CommandAck>) => void;
}) {
  const action = useConfirmedAction<CommandAck>(["plc"], { onSuccess });
  return (
    <>
      <LastActionPanel title="Last command" last={action.last} />
      <button
        type="button"
        onClick={() => {
          action.ask({
            title: "Trip the unit",
            body: <p>The fuel is shut off.</p>,
            confirmLabel: "Trip now",
            danger: true,
            run,
            done: "Tripped.",
          });
        }}
      >
        Trip…
      </button>
      <button
        type="button"
        onClick={() => {
          action.runNow("Pause", run);
        }}
      >
        Pause
      </button>
      {action.dialog}
    </>
  );
}

function renderHarness(run: () => Promise<CommandAck>, onSuccess?: () => void) {
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  render(
    <QueryClientProvider client={client}>
      <Harness run={run} onSuccess={onSuccess} />
    </QueryClientProvider>,
  );
  return { invalidate };
}

function glyphOf(panel: HTMLElement): string {
  return panel.querySelector("h2 svg")?.getAttribute("class") ?? "";
}

afterEach(() => {
  cleanup();
});

describe("useConfirmedAction", () => {
  it("runs a confirmed command once, invalidates what it changed and shows the answer", async () => {
    const run = vi.fn(() => Promise.resolve(accepted));
    const onSuccess = vi.fn();
    const { invalidate } = renderHarness(run, onSuccess);
    expect(screen.queryByRole("region", { name: "Last command" })).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Trip…" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Trip now" }),
    );
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    expect(run).toHaveBeenCalledTimes(1);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["plc"] });
    expect(onSuccess).toHaveBeenCalledWith(expect.objectContaining({ done: "Tripped." }));
    const last = screen.getByRole("region", { name: "Last command" });
    expect(last.textContent).toContain("Trip the unit");
    expect(within(last).getByRole("status").textContent).toBe("Accepted.");
    expect(glyphOf(last)).toContain("lucide-circle-check");
  });

  it("runs nothing when the operator cancels", async () => {
    const run = vi.fn(() => Promise.resolve(accepted));
    renderHarness(run);
    await userEvent.click(screen.getByRole("button", { name: "Trip…" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Cancel" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(run).not.toHaveBeenCalled();
    expect(screen.queryByRole("region", { name: "Last command" })).toBeNull();
  });

  it("runs an action that needs no confirmation at once", async () => {
    const run = vi.fn(() => Promise.resolve(accepted));
    renderHarness(run);
    await userEvent.click(screen.getByRole("button", { name: "Pause" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(run).toHaveBeenCalledTimes(1);
    expect((await screen.findByRole("region", { name: "Last command" })).textContent).toContain(
      "Pause",
    );
  });

  it("shows a gateway error with the refused glyph, not a tick", async () => {
    const run = vi.fn(() =>
      Promise.reject(
        new ApiError({
          status: 409,
          code: "plc.unavailable",
          title: "Conflict",
          detail: "The PLC did not answer.",
          errors: [],
          retryAfterS: null,
        }),
      ),
    );
    const onSuccess = vi.fn();
    renderHarness(run, onSuccess);
    await userEvent.click(screen.getByRole("button", { name: "Trip…" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Trip now" }),
    );
    const last = await screen.findByRole("region", { name: "Last command" });
    await waitFor(() => {
      expect(within(last).getByRole("alert").textContent).toBe("The PLC did not answer.");
    });
    expect(glyphOf(last)).toContain("lucide-circle-x");
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("shows a refusal of the PLC with the refused glyph", async () => {
    const run = vi.fn(() =>
      Promise.resolve({ accepted: false, reason: "Emergency stop is active.", timestamp_ms: 1 }),
    );
    renderHarness(run);
    await userEvent.click(screen.getByRole("button", { name: "Pause" }));
    const last = await screen.findByRole("region", { name: "Last command" });
    await waitFor(() => {
      expect(within(last).getByRole("alert").textContent).toBe(
        "Refused: Emergency stop is active.",
      );
    });
    expect(glyphOf(last)).toContain("lucide-circle-x");
  });
});
