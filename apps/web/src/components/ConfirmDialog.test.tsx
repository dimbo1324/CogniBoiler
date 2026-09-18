import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/http";
import { CommandResult } from "./CommandResult";
import { ConfirmDialog } from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  afterEach(() => {
    cleanup();
  });

  function renderDialog(busy = false) {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <ConfirmDialog
        title="Change the load demand"
        confirmLabel="Set 300.0 MW"
        busy={busy}
        onConfirm={onConfirm}
        onCancel={onCancel}
      >
        <p>From 250.0 MW to 300.0 MW.</p>
      </ConfirmDialog>,
    );
    return { onConfirm, onCancel };
  }

  it("says what will change and focuses the confirmation", () => {
    renderDialog();
    expect(screen.getByRole("dialog", { name: "Change the load demand" }).textContent).toContain(
      "From 250.0 MW to 300.0 MW.",
    );
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Set 300.0 MW" }));
  });

  it("confirms only on the confirmation button", async () => {
    const { onConfirm, onCancel } = renderDialog();
    await userEvent.click(screen.getByRole("button", { name: "Set 300.0 MW" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("cancels on the cancel button and on Escape", async () => {
    const { onConfirm, onCancel } = renderDialog();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await userEvent.keyboard("{Escape}");
    expect(onCancel).toHaveBeenCalledTimes(2);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("cannot be confirmed twice while the command is on its way", () => {
    renderDialog(true);
    expect(
      (screen.getByRole("button", { name: "Set 300.0 MW" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("CommandResult", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows the PLC's refusal with its reason", () => {
    render(
      <CommandResult
        result={{ accepted: false, reason: "Emergency stop is active." }}
        error={null}
      />,
    );
    expect(screen.getByRole("alert").textContent).toBe("Refused: Emergency stop is active.");
  });

  it("shows an acceptance", () => {
    render(<CommandResult result={{ accepted: true, reason: "" }} error={null} />);
    expect(screen.getByRole("status").textContent).toBe("Accepted.");
  });

  it("shows the gateway's refusal before any result", () => {
    const error = new ApiError({
      status: 403,
      code: "auth.forbidden",
      title: "Forbidden",
      detail: "Requires the engineer role.",
      errors: [],
      retryAfterS: null,
    });
    render(<CommandResult result={{ accepted: true, reason: "" }} error={error} />);
    expect(screen.getByRole("alert").textContent).toBe("Requires the engineer role.");
  });
});
