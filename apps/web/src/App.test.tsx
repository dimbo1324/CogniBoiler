import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { fetchHealth } from "./api/client";

vi.mock("./api/client", () => ({ fetchHealth: vi.fn() }));

const mockedFetchHealth = vi.mocked(fetchHealth);

describe("App", () => {
  afterEach(() => {
    cleanup();
    vi.resetAllMocks();
  });

  it("reports the gateway status and version once the health check answers", async () => {
    mockedFetchHealth.mockResolvedValue({
      service: "CogniBoiler API Gateway",
      version: "0.1.0",
      status: "running",
    });

    render(<App />);

    expect(await screen.findByText("API gateway running (version 0.1.0)")).toBeDefined();
  });

  it("says why the gateway is unreachable", async () => {
    mockedFetchHealth.mockRejectedValue(new Error("connection refused"));

    render(<App />);

    expect(await screen.findByText("API gateway unreachable: connection refused")).toBeDefined();
  });
});
