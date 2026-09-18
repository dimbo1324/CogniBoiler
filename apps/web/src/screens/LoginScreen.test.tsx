import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/http";
import { SessionProvider } from "../session/SessionProvider";
import { SessionManager } from "../session/session";
import { LoginScreen, signInError } from "./LoginScreen";

function apiError(status: number, code: string, retryAfterS: number | null = null): ApiError {
  return new ApiError({
    status,
    code,
    title: "Error",
    detail: "The gateway said no.",
    errors: [],
    retryAfterS,
  });
}

function renderLogin(manager: SessionManager, notice: string | null = null) {
  vi.spyOn(manager, "restore").mockResolvedValue();
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <SessionProvider manager={manager}>
        <LoginScreen notice={notice} />
      </SessionProvider>
    </QueryClientProvider>,
  );
}

describe("LoginScreen", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("signs in with the trimmed username and the password as typed", async () => {
    const manager = new SessionManager();
    const signIn = vi.spyOn(manager, "signIn").mockResolvedValue();
    renderLogin(manager);

    await userEvent.type(screen.getByLabelText("Username"), " operator ");
    await userEvent.type(screen.getByLabelText("Password"), " pass word ");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(signIn).toHaveBeenCalledWith("operator", " pass word ");
  });

  it("shows why sign-in failed and clears the password", async () => {
    const manager = new SessionManager();
    vi.spyOn(manager, "signIn").mockRejectedValue(apiError(401, "auth.invalid_credentials"));
    renderLogin(manager);

    await userEvent.type(screen.getByLabelText("Username"), "operator");
    await userEvent.type(screen.getByLabelText("Password"), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toBe("Invalid username or password.");
    expect((screen.getByLabelText("Password") as HTMLInputElement).value).toBe("");
  });

  it("shows why the previous session ended", () => {
    renderLogin(new SessionManager(), "Your session has ended. Sign in again.");
    expect(screen.getByRole("status").textContent).toBe("Your session has ended. Sign in again.");
  });
});

describe("signInError", () => {
  it("never tells an unknown account from a wrong password", () => {
    expect(signInError(apiError(401, "auth.invalid_credentials"))).toBe(
      "Invalid username or password.",
    );
  });

  it("says how long to wait after too many attempts", () => {
    expect(signInError(apiError(429, "auth.too_many_attempts", 610))).toBe(
      "Too many failed sign-in attempts. Try again in 11 min.",
    );
  });

  it("explains an unreachable gateway and a malformed form", () => {
    expect(signInError(apiError(0, "network.unreachable"))).toBe("The gateway cannot be reached.");
    expect(signInError(apiError(422, "request.invalid"))).toMatch(/3–64 characters/u);
    expect(signInError(new Error("boom"))).toBe("Sign-in failed.");
  });
});
