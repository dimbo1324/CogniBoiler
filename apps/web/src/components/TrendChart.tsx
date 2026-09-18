import { useEffect, useRef } from "react";
import uPlot from "uplot";

import type { TrendParameter } from "../trends/parameters";

const HEIGHT_PX = 150;

function cssVariable(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** One parameter over time. Charts that share `syncKey` share the cursor. */
export function TrendChart({
  parameter,
  times,
  values,
  syncKey,
}: {
  parameter: TrendParameter;
  times: number[];
  values: (number | null)[];
  syncKey: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);

  useEffect(() => {
    const element = container.current;
    if (element === null) {
      return;
    }
    const text = cssVariable("--text", "#1c2027");
    const grid = cssVariable("--border", "#d3d7de");
    const options: uPlot.Options = {
      width: Math.max(element.clientWidth, 200),
      height: HEIGHT_PX,
      cursor: { sync: { key: syncKey } },
      legend: { show: true },
      scales: { x: { time: true } },
      axes: [
        { stroke: text, grid: { stroke: grid } },
        { stroke: text, grid: { stroke: grid }, size: 64 },
      ],
      series: [
        {},
        {
          label: `${parameter.label} [${parameter.unit}]`,
          stroke: cssVariable("--accent", "#1f6feb"),
          width: 1.5,
          spanGaps: true,
          value: (_, value) => (value === null ? "—" : value.toFixed(parameter.digits)),
        },
      ],
    };
    const instance = new uPlot(options, [[], []], element);
    chart.current = instance;
    const observer = new ResizeObserver(() => {
      instance.setSize({ width: Math.max(element.clientWidth, 200), height: HEIGHT_PX });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      instance.destroy();
      chart.current = null;
    };
  }, [parameter, syncKey]);

  useEffect(() => {
    chart.current?.setData([times, values] as uPlot.AlignedData);
  }, [times, values]);

  return (
    <div
      ref={container}
      className="trend-chart"
      role="img"
      aria-label={`${parameter.label} trend in ${parameter.unit}`}
      data-testid={`trend-${parameter.id}`}
    />
  );
}
