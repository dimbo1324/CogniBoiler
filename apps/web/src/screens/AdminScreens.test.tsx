import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createUser,
  fetchAudit,
  fetchUsers,
  resetUserPassword,
  revokeUserSessions,
  updateUser,
} from "../api/endpoints";
import { ApiError } from "../api/http";
import type { AuditEntry, User } from "../api/types";
import { AuditScreen } from "./AuditScreen";
import { UsersScreen } from "./UsersScreen";

vi.mock("../api/endpoints", async (importOriginal) => {
  const original = await importOriginal<typeof import("../api/endpoints")>();
  return {
    ...original,
    fetchAudit: vi.fn(),
    fetchUsers: vi.fn(),
    createUser: vi.fn(),
    updateUser: vi.fn(),
    resetUserPassword: vi.fn(),
    revokeUserSessions: vi.fn(),
  };
});

vi.mock("../session/SessionProvider", () => ({
  useUser: () => ({ username: "admin", role: "admin" }),
}));

function renderScreen(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

function entry(overrides: Partial<AuditEntry> = {}): AuditEntry {
  return {
    id: 1,
    user_id: 2,
    username: "operator1",
    role: "operator",
    ip_address: "10.0.0.9",
    method: "POST",
    endpoint: "/api/v1/commands/load",
    request_body_hash: null,
    response_status: 200,
    duration_ms: 12,
    timestamp_ms: 1_789_700_000_000,
    detail: "load_w=180000000",
    outcome: "accepted",
    ...overrides,
  } as AuditEntry;
}

function user(overrides: Partial<User> = {}): User {
  return {
    id: 2,
    username: "operator1",
    role: "operator",
    is_active: true,
    created_at_ms: 1,
    last_login_at_ms: null,
    open_sessions: 1,
    ...overrides,
  };
}

const PASSWORD = "Harbour-Lantern-42";

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("AuditScreen", () => {
  beforeEach(() => {
    vi.mocked(fetchAudit).mockResolvedValue({
      items: [entry(), entry({ id: 2, response_status: 403, outcome: "refused", detail: null })],
      total: 120,
      limit: 50,
      offset: 0,
    });
  });

  it("lists entries with who, what, outcome and where from", async () => {
    renderScreen(<AuditScreen />);
    await screen.findByText("accepted");
    const row = screen.getAllByRole("row")[1];
    const cells = within(row as HTMLElement).getAllByRole("cell");
    expect(cells.map((cell) => cell.textContent)).toEqual([
      expect.stringMatching(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC/u) as string,
      "operator1",
      "operator",
      "POST /api/v1/commands/loadload_w=180000000",
      "200",
      "accepted",
      "12 ms",
      "10.0.0.9",
    ]);
    expect(screen.getAllByRole("row")[2]?.className).toBe("severity-warning");
  });

  it("applies the filters as one query from the first page", async () => {
    renderScreen(<AuditScreen />);
    await screen.findByText("accepted");
    await userEvent.type(screen.getByLabelText("User"), " operator1 ");
    await userEvent.selectOptions(screen.getByLabelText("Method"), "POST");
    await userEvent.type(screen.getByLabelText("Path starts with"), "/api/v1/commands");
    await userEvent.click(screen.getByLabelText("Refusals only"));
    await userEvent.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => {
      expect(vi.mocked(fetchAudit)).toHaveBeenLastCalledWith(
        {
          username: "operator1",
          method: "POST",
          endpoint: "/api/v1/commands",
          minStatus: 400,
          fromMs: null,
          toMs: null,
          limit: 50,
          offset: 0,
        },
        expect.anything(),
      );
    });
  });

  it("pages through the log", async () => {
    renderScreen(<AuditScreen />);
    await screen.findByText("1–50 of 120");
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => {
      expect(vi.mocked(fetchAudit).mock.lastCall?.[0].offset).toBe(50);
    });
  });

  it("explains a refusal of the gateway", async () => {
    vi.mocked(fetchAudit).mockRejectedValue(
      new ApiError({
        status: 403,
        code: "auth.forbidden",
        title: "Forbidden",
        detail: "This action requires the admin role or higher.",
        errors: [],
        retryAfterS: null,
      }),
    );
    renderScreen(<AuditScreen />);
    expect(await screen.findByText("This action requires the admin role or higher.")).toBeDefined();
  });
});

describe("UsersScreen", () => {
  beforeEach(() => {
    vi.mocked(fetchUsers).mockResolvedValue({
      items: [
        user({ id: 1, username: "admin", role: "admin" }),
        user(),
        user({ id: 3, username: "viewer1", role: "viewer", open_sessions: 0 }),
      ],
      total: 3,
      limit: 50,
      offset: 0,
    });
  });

  it("protects the signed-in administrator's own row", async () => {
    renderScreen(<UsersScreen />);
    const own = await screen.findByTestId("user-admin");
    expect(own.textContent).toContain("(you)");
    expect((within(own).getByLabelText("Role of admin") as HTMLSelectElement).disabled).toBe(true);
    expect(
      (within(own).getByRole("button", { name: "Block…" }) as HTMLButtonElement).disabled,
    ).toBe(true);
    const quiet = screen.getByTestId("user-viewer1");
    const signOut = within(quiet).getByRole("button", { name: "Sign out everywhere…" });
    expect((signOut as HTMLButtonElement).disabled).toBe(true);
  });

  it("creates an account only with a valid name and a long password", async () => {
    vi.mocked(createUser).mockResolvedValue(
      user({ id: 9, username: "shift.lead", role: "engineer" }),
    );
    renderScreen(<UsersScreen />);
    const form = await screen.findByRole("form", { name: "Create a user" });
    const create = within(form).getByRole("button", { name: "Create" }) as HTMLButtonElement;
    await userEvent.type(within(form).getByLabelText("Username"), "shift.lead");
    await userEvent.type(within(form).getByLabelText(/Initial password/u), "short");
    expect(create.disabled).toBe(true);
    await userEvent.clear(within(form).getByLabelText(/Initial password/u));
    await userEvent.type(within(form).getByLabelText(/Initial password/u), PASSWORD);
    await userEvent.selectOptions(within(form).getByLabelText("Role"), "engineer");
    expect(create.disabled).toBe(false);
    await userEvent.click(create);
    expect(vi.mocked(createUser)).toHaveBeenCalledWith({
      username: "shift.lead",
      password: PASSWORD,
      role: "engineer",
    });
    expect((await screen.findByRole("status")).textContent).toBe("Created shift.lead as engineer.");
    expect((within(form).getByLabelText("Username") as HTMLInputElement).value).toBe("");
  });

  it("refuses a name outside the gateway's pattern before sending", async () => {
    renderScreen(<UsersScreen />);
    const form = await screen.findByRole("form", { name: "Create a user" });
    await userEvent.type(within(form).getByLabelText("Username"), "has space");
    await userEvent.type(within(form).getByLabelText(/Initial password/u), PASSWORD);
    expect(
      (within(form).getByRole("button", { name: "Create" }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("changes a role after confirmation", async () => {
    vi.mocked(updateUser).mockResolvedValue(user({ role: "engineer" }));
    renderScreen(<UsersScreen />);
    const row = await screen.findByTestId("user-operator1");
    await userEvent.selectOptions(within(row).getByLabelText("Role of operator1"), "engineer");
    await userEvent.click(within(row).getByRole("button", { name: "Apply…" }));
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("From operator to engineer");
    await userEvent.click(within(dialog).getByRole("button", { name: "Make engineer" }));
    expect(vi.mocked(updateUser)).toHaveBeenCalledWith(2, { role: "engineer" });
    expect((await screen.findByRole("status")).textContent).toBe("operator1 is now engineer.");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("blocks a user after confirmation, and a cancel sends nothing", async () => {
    vi.mocked(updateUser).mockResolvedValue(user({ is_active: false }));
    renderScreen(<UsersScreen />);
    const row = await screen.findByTestId("user-operator1");
    await userEvent.click(within(row).getByRole("button", { name: "Block…" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(vi.mocked(updateUser)).not.toHaveBeenCalled();
    await userEvent.click(within(row).getByRole("button", { name: "Block…" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Block" }),
    );
    expect(vi.mocked(updateUser)).toHaveBeenCalledWith(2, { is_active: false });
  });

  it("resets a password only once it is long enough", async () => {
    vi.mocked(resetUserPassword).mockResolvedValue({ message: "Password reset." });
    renderScreen(<UsersScreen />);
    const row = await screen.findByTestId("user-operator1");
    const reset = within(row).getByRole("button", { name: "Reset password…" }) as HTMLButtonElement;
    const field = within(row).getByLabelText("New password for operator1");
    await userEvent.type(field, "too-short");
    expect(reset.disabled).toBe(true);
    await userEvent.clear(field);
    await userEvent.type(field, PASSWORD);
    await userEvent.click(reset);
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Reset password" }),
    );
    expect(vi.mocked(resetUserPassword)).toHaveBeenCalledWith(2, PASSWORD);
    expect((await screen.findByRole("status")).textContent).toBe(
      "The password of operator1 was reset.",
    );
  });

  it("signs a user out everywhere and shows a refusal", async () => {
    vi.mocked(revokeUserSessions).mockRejectedValue(
      new ApiError({
        status: 404,
        code: "users.not_found",
        title: "Not Found",
        detail: "User 2 does not exist.",
        errors: [],
        retryAfterS: null,
      }),
    );
    renderScreen(<UsersScreen />);
    const row = await screen.findByTestId("user-operator1");
    await userEvent.click(within(row).getByRole("button", { name: "Sign out everywhere…" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Sign out everywhere" }),
    );
    expect(vi.mocked(revokeUserSessions)).toHaveBeenCalledWith(2);
    expect((await screen.findByRole("alert")).textContent).toBe("User 2 does not exist.");
  });
});
