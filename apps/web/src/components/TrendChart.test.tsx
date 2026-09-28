import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { trendParameter, type TrendParameter } from "../trends/parameters";
import { TrendChart } from "./TrendChart";

const charts = vi.hoisted(() => ({ strokes: [] as string[], destroyed: 0 }));

vi.mock("uplot", () => ({
  default: class {
    constructor(options: { axes: { stroke: string }[] }) {
      charts.strokes.push(options.axes[0]?.stroke ?? "");
    }
    setData() {
      return undefined;
    }
    setSize() {
      return undefined;
    }
    destroy() {
      charts.destroyed += 1;
    }
  },
}));

class NoResize {
  observe() {
    return undefined;
  }
  disconnect() {
    return undefined;
  }
}

function power(): TrendParameter {
  const parameter = trendParameter("electrical_power");
  if (parameter === undefined) {
    throw new Error("electrical_power is not a trend parameter");
  }
  return parameter;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.style.removeProperty("--text");
});

describe("TrendChart", () => {
  it("redraws with the new theme's colours when the theme changes", async () => {
    vi.stubGlobal("ResizeObserver", NoResize);
    charts.strokes.length = 0;
    charts.destroyed = 0;
    document.documentElement.setAttribute("data-theme", "dark");
    document.documentElement.style.setProperty("--text", "#eeeeee");
    render(<TrendChart parameter={power()} times={[]} values={[]} syncKey="trends" />);
    expect(charts.strokes).toEqual(["#eeeeee"]);

    await act(async () => {
      document.documentElement.style.setProperty("--text", "#111111");
      document.documentElement.setAttribute("data-theme", "light");
      await Promise.resolve();
    });
    expect(charts.destroyed).toBe(1);
    expect(charts.strokes).toEqual(["#eeeeee", "#111111"]);
  });
});
