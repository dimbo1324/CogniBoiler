import { useQuery } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

import { fetchAudit, type AuditFilter } from "../api/endpoints";
import { ErrorOf } from "../components/ui/Note";
import { AuditIcon } from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";
import { Pager } from "../components/Pager";
import { formatDateTime, localInputToMs } from "../units";
import { queryKeys } from "../api/queryKeys";

const PAGE = 50;
const METHODS = ["", "GET", "POST", "PATCH", "DELETE", "WS"] as const;

export function AuditScreen() {
  const [draft, setDraft] = useState({
    username: "",
    method: "",
    endpoint: "",
    refusals: false,
    from: "",
    to: "",
  });
  const [filter, setFilter] = useState<AuditFilter>({ limit: PAGE, offset: 0 });
  const page = useQuery({
    queryKey: queryKeys.audit(filter),
    queryFn: ({ signal }) => fetchAudit(filter, signal),
  });

  const apply = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFilter({
      username: draft.username.trim(),
      method: draft.method,
      endpoint: draft.endpoint.trim(),
      minStatus: draft.refusals ? 400 : null,
      fromMs: localInputToMs(draft.from),
      toMs: localInputToMs(draft.to),
      limit: PAGE,
      offset: 0,
    });
  };

  return (
    <Panel title="Audit log" glyph={AuditIcon}>
      <p className="muted">
        Every sign-in, sign-out, refusal and change, newest first. The log is append-only: the
        database refuses to change or delete an entry.
      </p>
      <form className="form-grid" onSubmit={apply} aria-label="Audit filters">
        <label>
          User
          <input
            value={draft.username}
            onChange={(event) => {
              setDraft({ ...draft, username: event.target.value });
            }}
          />
        </label>
        <label>
          Method
          <select
            value={draft.method}
            onChange={(event) => {
              setDraft({ ...draft, method: event.target.value });
            }}
          >
            {METHODS.map((method) => (
              <option key={method} value={method}>
                {method || "any"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Path starts with
          <input
            value={draft.endpoint}
            placeholder="/api/v1/commands"
            onChange={(event) => {
              setDraft({ ...draft, endpoint: event.target.value });
            }}
          />
        </label>
        <label>
          From
          <input
            type="datetime-local"
            step={1}
            value={draft.from}
            onChange={(event) => {
              setDraft({ ...draft, from: event.target.value });
            }}
          />
        </label>
        <label>
          To
          <input
            type="datetime-local"
            step={1}
            value={draft.to}
            onChange={(event) => {
              setDraft({ ...draft, to: event.target.value });
            }}
          />
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={draft.refusals}
            onChange={(event) => {
              setDraft({ ...draft, refusals: event.target.checked });
            }}
          />
          Refusals only
        </label>
        <button type="submit">Apply</button>
      </form>
      {page.isError && <ErrorOf error={page.error} />}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>At</th>
              <th>User</th>
              <th>Role</th>
              <th>Request</th>
              <th className="number">Status</th>
              <th>Outcome</th>
              <th className="number">Took</th>
              <th>From</th>
            </tr>
          </thead>
          <tbody>
            {(page.data?.items ?? []).map((entry) => (
              <tr key={entry.id} className={entry.response_status >= 400 ? "severity-warning" : ""}>
                <td>{formatDateTime(entry.timestamp_ms)}</td>
                <td>{entry.username ?? "—"}</td>
                <td>{entry.role ?? "—"}</td>
                <td>
                  {entry.method} {entry.endpoint}
                  {entry.detail && <div className="muted">{entry.detail}</div>}
                </td>
                <td className="number">{entry.response_status}</td>
                <td>{entry.outcome ?? "—"}</td>
                <td className="number">{entry.duration_ms} ms</td>
                <td>{entry.ip_address}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager
        offset={filter.offset}
        limit={filter.limit}
        total={page.data?.total ?? 0}
        onChange={(offset) => {
          setFilter({ ...filter, offset });
        }}
      />
    </Panel>
  );
}
