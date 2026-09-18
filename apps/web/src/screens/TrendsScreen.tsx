import { useQueries, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { fetchHistory, fetchKpi } from "../api/endpoints";
import type { HistoryMeasurement, HistoryResponse, Kpi } from "../api/types";
import { TrendChart } from "../components/TrendChart";
import { useLive, useLiveStore } from "../live/LiveProvider";
import { DEFAULT_TREND_IDS, TREND_PARAMETERS, type TrendParameter } from "../trends/parameters";
import { buildColumns } from "../trends/series";
import {
  co2PerJouleToKilogramsPerMegawattHour,
  formatDateTime,
  formatReading,
  fractionToPercent,
  heatRateToKilojoulesPerKilowattHour,
  wattsToMegawatts,
} from "../units";

export type TrendRange = "live" | "15m" | "1h" | "24h";

const RANGES: readonly { id: TrendRange; label: string; spanMs: number }[] = [
  { id: "live", label: "Live", spanMs: 15 * 60_000 },
  { id: "15m", label: "15 min", spanMs: 15 * 60_000 },
  { id: "1h", label: "1 h", spanMs: 60 * 60_000 },
  { id: "24h", label: "24 h", spanMs: 24 * 60 * 60_000 },
];

// Points per history request: enough for a smooth line at laptop width.
const HISTORY_POINTS = 600;
const LIVE_KPI_REFRESH_MS = 30_000;

/** Fields to read per measurement for the chosen parameters. */
export function historyRequests(
  parameters: readonly TrendParameter[],
): { measurement: HistoryMeasurement; fields: string[] }[] {
  const fields = new Map<HistoryMeasurement, string[]>();
  for (const parameter of parameters) {
    const list = fields.get(parameter.history.measurement) ?? [];
    list.push(parameter.history.field);
    fields.set(parameter.history.measurement, list);
  }
  return [...fields].map(([measurement, list]) => ({ measurement, fields: list }));
}

function KpiPanel({ kpi, isError }: { kpi: Kpi | undefined; isError: boolean }) {
  if (isError) {
    return <p className="error">KPIs are unavailable: the historian did not answer.</p>;
  }
  if (!kpi) {
    return <p className="muted">Loading KPIs…</p>;
  }
  const items: [string, string][] = [
    [
      "Mean output",
      `${formatReading(kpi.mean_electrical_power_w === null ? null : wattsToMegawatts(kpi.mean_electrical_power_w), 1)} MW`,
    ],
    [
      "Net efficiency",
      `${formatReading(kpi.net_efficiency === null ? null : fractionToPercent(kpi.net_efficiency), 2)} %`,
    ],
    [
      "Boiler efficiency",
      `${formatReading(kpi.boiler_efficiency === null ? null : fractionToPercent(kpi.boiler_efficiency), 2)} %`,
    ],
    [
      "Plant heat rate",
      `${formatReading(kpi.plant_heat_rate_j_per_j === null ? null : heatRateToKilojoulesPerKilowattHour(kpi.plant_heat_rate_j_per_j), 0)} kJ/kWh`,
    ],
    [
      "Turbine heat rate",
      `${formatReading(kpi.turbine_heat_rate_j_per_j === null ? null : heatRateToKilojoulesPerKilowattHour(kpi.turbine_heat_rate_j_per_j), 0)} kJ/kWh`,
    ],
    [
      "CO2 intensity",
      `${formatReading(kpi.co2_intensity_kg_per_j === null ? null : co2PerJouleToKilogramsPerMegawattHour(kpi.co2_intensity_kg_per_j), 0)} kg/MWh`,
    ],
    [
      "NOx mean / peak",
      `${formatReading(kpi.mean_nox_ppmv, 1)} / ${formatReading(kpi.peak_nox_ppmv, 1)} ppmv`,
    ],
    [
      "Health mean / lowest",
      `${formatReading(kpi.mean_health_pct, 1)} / ${formatReading(kpi.lowest_health_pct, 1)} %`,
    ],
  ];
  return (
    <>
      <div className="kpis" data-testid="kpis">
        {items.map(([label, value]) => (
          <div className="kpi" key={label}>
            <div className="muted">{label}</div>
            <div className="kpi-value">{value}</div>
          </div>
        ))}
      </div>
      <p className="muted">
        {kpi.samples} samples from {kpi.source === "raw" ? "raw data" : "one-minute aggregates"},{" "}
        {formatDateTime(kpi.start_ms)} – {formatDateTime(kpi.end_ms)}. Ratios are ratios of means
        over the range.
      </p>
    </>
  );
}

export function TrendsScreen() {
  const [range, setRange] = useState<TrendRange>("live");
  const [selected, setSelected] = useState<readonly string[]>(DEFAULT_TREND_IDS);
  const [anchorMs, setAnchorMs] = useState(() => Date.now());
  const live = useLive();
  const store = useLiveStore();

  const spanMs = RANGES.find((item) => item.id === range)?.spanMs ?? 15 * 60_000;
  const parameters = useMemo(
    () => TREND_PARAMETERS.filter((parameter) => selected.includes(parameter.id)),
    [selected],
  );
  const requests = useMemo(() => historyRequests(parameters), [parameters]);
  const startMs = anchorMs - spanMs;

  const history = useQueries({
    queries: requests.map((item) => ({
      queryKey: ["history", item.measurement, item.fields.join(","), range, anchorMs],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        fetchHistory(item.measurement, item.fields, startMs, anchorMs, HISTORY_POINTS, signal),
      enabled: range !== "live",
      // Recorded history of a fixed range does not change; the live stream continues it.
      staleTime: Number.POSITIVE_INFINITY,
    })),
  });
  const kpi = useQuery({
    queryKey: ["kpi", range, anchorMs],
    queryFn: ({ signal }) => {
      const endMs = range === "live" ? Date.now() : anchorMs;
      return fetchKpi(endMs - spanMs, endMs, signal);
    },
    refetchInterval: range === "live" ? LIVE_KPI_REFRESH_MS : false,
  });

  const historyData = history
    .map((result) => result.data)
    .filter((data): data is HistoryResponse => data !== undefined);
  const historyFailed = history.some((result) => result.isError);
  // Rebuilt on every live sample (useLive re-renders twice a second); a few thousand
  // points cost far less than a frame.
  const liveSinceMs = range === "live" ? (live.plantReceivedAtMs ?? anchorMs) - spanMs : startMs;
  const columns = buildColumns(
    parameters,
    range === "live" ? [] : historyData,
    store.samples.since(liveSinceMs),
    liveSinceMs,
  );

  return (
    <div className="stack">
      <section className="panel" aria-label="Trend settings">
        <div className="row" role="group" aria-label="Time range">
          {RANGES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === range ? "primary" : ""}
              aria-pressed={item.id === range}
              onClick={() => {
                setRange(item.id);
                setAnchorMs(Date.now());
              }}
            >
              {item.label}
            </button>
          ))}
          {range !== "live" && (
            <button
              type="button"
              onClick={() => {
                setAnchorMs(Date.now());
              }}
            >
              Reload history
            </button>
          )}
        </div>
        <fieldset className="trend-picker">
          <legend>Parameters</legend>
          {TREND_PARAMETERS.map((parameter) => (
            <label key={parameter.id}>
              <input
                type="checkbox"
                checked={selected.includes(parameter.id)}
                onChange={(event) => {
                  const checked = event.target.checked;
                  setSelected((current) =>
                    checked
                      ? [...current, parameter.id]
                      : current.filter((id) => id !== parameter.id),
                  );
                }}
              />{" "}
              {parameter.label} [{parameter.unit}]
            </label>
          ))}
        </fieldset>
        {range === "live" ? (
          <p className="muted">Live: the last 15 minutes received on this page, twice a second.</p>
        ) : (
          <p className="muted">
            Recorded history from {formatDateTime(startMs)}, continued by the live stream.
            {historyFailed && (
              <span className="error"> History is unavailable: the historian did not answer.</span>
            )}
          </p>
        )}
      </section>
      <section className="panel trend-list" aria-label="Trends">
        {parameters.length === 0 && <p className="muted">Choose at least one parameter.</p>}
        {parameters.map((parameter, index) => (
          <TrendChart
            key={parameter.id}
            parameter={parameter}
            times={columns.times}
            values={columns.values[index] ?? []}
            syncKey="trends"
          />
        ))}
      </section>
      <section className="panel" aria-label="KPIs">
        <h2>KPIs for the range</h2>
        <KpiPanel kpi={kpi.data} isError={kpi.isError} />
      </section>
    </div>
  );
}
