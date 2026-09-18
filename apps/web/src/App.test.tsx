import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { ApiError } from "./api/http";
import { refreshSession } from "./api/endpoints";

vi.mock("./api/endpoints", () => ({ refreshSession: vi.fn() }));

const mockedRefresh = vi.mocked(refreshSession);

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App", () => {
  afterEach(() => {
    cleanup();
    vi.resetAllMocks();
  });

  it("says it is restoring the session, then asks to sign in when there is none", async () => {
    mockedRefresh.mockRejectedValue(
      new ApiError({
        status: 401,
        code: "auth.refresh_missing",
        title: "Unauthorized",
        detail: "No refresh token was presented.",
        errors: [],
        retryAfterS: null,
      }),
    );

    renderApp();

    expect(screen.getByRole("status").textContent).toBe("Restoring your session…");
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeDefined();
    expect(mockedRefresh).toHaveBeenCalledTimes(1);
  });

  it("tells the operator when the gateway cannot be reached", async () => {
    mockedRefresh.mockRejectedValue(
      new ApiError({
        status: 0,
        code: "network.unreachable",
        title: "Network error",
        detail: "The gateway cannot be reached.",
        errors: [],
        retryAfterS: null,
      }),
    );

    renderApp();

    expect(await screen.findByText(/The gateway cannot be reached/u)).toBeDefined();
  });
});
