import { useQuery } from "@tanstack/react-query";

import { fetchPlatform } from "../api/endpoints";
import { describeError } from "../api/http";
import { useLive } from "../live/LiveProvider";
import { formatDateTime, formatReading } from "../units";

const REFRESH_MS = 5_000;

const STATUS_TEXT = {
  ready: "Ready: the database and every service answer.",
  degraded: "Degraded: the gateway serves, but a service is down.",
  not_ready: "Not ready: the database is down, so nobody can be authenticated.",
} as const;

export function PlatformScreen() {
  const live = useLive();
  const platform = useQuery({
    queryKey: ["platform"],
    queryFn: ({ signal }) => fetchPlatform(signal),
    refetchInterval: REFRESH_MS,
  });
  const data = platform.data;
  return (
    <div className="stack">
      <section className="panel" aria-label="Services">
        <h2>Services</h2>
        {platform.isError && <p className="error">{describeError(platform.error)}</p>}
        {data && (
          <>
            <p data-testid="platform-status">{STATUS_TEXT[data.readiness.status]}</p>
            <table>
              <thead>
                <tr>
                  <th>Component</th>
                  <th>State</th>
                  <th>Required</th>
                  <th className="number">Round trip</th>
                </tr>
              </thead>
              <tbody>
                {data.readiness.components.map((component) => (
                  <tr
                    key={component.name}
                    className={component.state === "up" ? "" : "severity-critical"}
                    data-testid={`component-${component.name}`}
                  >
                    <td>{component.name}</td>
                    <td>{component.state}</td>
                    <td>{component.required ? "yes" : "no"}</td>
                    <td className="number">
                      {component.latency_ms === null
                        ? "—"
                        : `${formatReading(component.latency_ms, 1)} ms`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">Checked {formatDateTime(data.readiness.checked_at_ms)}.</p>
          </>
        )}
      </section>
      <section className="panel" aria-label="Live data">
        <h2>Live data</h2>
        <dl className="kv">
          <dt>This console</dt>
          <dd>{live.connection}</dd>
          <dt>Latest plant state at the gateway</dt>
          <dd>
            {data?.telemetry_age_s === null || data === undefined
              ? "—"
              : `${formatReading(data.telemetry_age_s, 1)} s ago`}
          </dd>
          <dt>Latest plant state on this page</dt>
          <dd>{formatDateTime(live.plantReceivedAtMs)}</dd>
          <dt>WebSocket clients</dt>
          <dd>{data?.websocket_clients ?? "—"}</dd>
        </dl>
      </section>
    </div>
  );
}
