import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RecommendationsPanel, insightsEnabled } from "./RecommendationsPanel";

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
});

describe("RecommendationsPanel", () => {
  it("shows nothing while no insight service is configured", () => {
    const { container } = render(<RecommendationsPanel />);
    expect(insightsEnabled()).toBe(false);
    expect(container.innerHTML).toBe("");
  });

  it("keeps its place ready for the deferred AI stage", () => {
    vi.stubEnv("VITE_INSIGHTS", "true");
    render(<RecommendationsPanel />);
    expect(insightsEnabled()).toBe(true);
    expect(screen.getByRole("region", { name: "Recommendations" })).toBeDefined();
    expect(screen.getByRole("status").textContent).toBe(
      "No recommendations: the insight service does not answer yet.",
    );
  });

  it("stays hidden for any other value", () => {
    vi.stubEnv("VITE_INSIGHTS", "1");
    const { container } = render(<RecommendationsPanel />);
    expect(container.innerHTML).toBe("");
  });
});
