import { useQuery } from "@tanstack/react-query";
import { useState, type FormEvent, type ReactNode } from "react";

import {
  isUnacknowledged,
  sortAlarms,
  useAcknowledge,
  useAcknowledgeAll,
  useActiveAlarms,
} from "../alarms/queries";
import { fetchAlarm, fetchAlarmHistory, type AlarmHistoryFilter } from "../api/endpoints";
import { describeError } from "../api/http";
import type { Alarm, AlarmState } from "../api/types";
import { Pager } from "../components/Pager";
import { useCan } from "../session/SessionProvider";
import { formatDateTime, formatQuantity, parameterLabel } from "../units";

const STATE_LABEL: Record<AlarmState, string> = {
  ACTIVE_UNACK: "active, unacknowledged",
  ACTIVE_ACK: "active, acknowledged",
  CLEARED_UNACK: "cleared, unacknowledged",
  CLEARED: "cleared",
};

const HISTORY_PAGE = 25;

function AlarmRow({
  alarm,
  selected,
  onSelect,
  action,
}: {
  alarm: Alarm;
  selected: boolean;
  onSelect: () => void;
  action?: ReactNode;
}) {
  const classes = [`severity-${alarm.severity}`];
  if (isUnacknowledged(alarm)) {
    classes.push("unacknowledged");
    if (alarm.severity === "critical") {
      classes.push("flashing");
    }
  }
  return (
    <tr
      className={classes.join(" ")}
      aria-selected={selected}
      data-testid={`alarm-${String(alarm.id)}`}
    >
      <td>{alarm.severity}</td>
      <td>{STATE_LABEL[alarm.state]}</td>
      <td>
        <button type="button" className="link" onClick={onSelect}>
          {parameterLabel(alarm.parameter)}
        </button>
        <div className="muted">{alarm.message}</div>
      </td>
      <td className="number">{formatQuantity(alarm.value, alarm.unit)}</td>
      <td className="number">{formatQuantity(alarm.threshold, alarm.unit)}</td>
      <td>{formatDateTime(alarm.raised_at_ms)}</td>
      <td>
        {alarm.acknowledged_by
          ? `${alarm.acknowledged_by}, ${formatDateTime(alarm.acknowledged_at_ms)}`
          : "—"}
      </td>
      {action !== undefined && <td>{action}</td>}
    </tr>
  );
}

function AlarmTableHead({ withAction }: { withAction: boolean }) {
  return (
    <thead>
      <tr>
        <th>Severity</th>
        <th>State</th>
        <th>Parameter</th>
        <th className="number">Value</th>
        <th className="number">Limit</th>
        <th>Raised</th>
        <th>Acknowledged</th>
        {withAction && <th>Action</th>}
      </tr>
    </thead>
  );
}

function AlarmDetailPanel({ alarmId }: { alarmId: number }) {
  const detail = useQuery({
    queryKey: ["alarms", "detail", alarmId],
    queryFn: ({ signal }) => fetchAlarm(alarmId, signal),
  });
  if (detail.isError) {
    return <p className="error">{describeError(detail.error)}</p>;
  }
  if (!detail.data) {
    return <p className="muted">Loading…</p>;
  }
  const { alarm, transitions } = detail.data;
  return (
    <section className="panel" aria-label="Alarm details">
      <h2>
        Alarm {alarm.id}: {parameterLabel(alarm.parameter)} ({alarm.severity})
      </h2>
      <p>{alarm.message}</p>
      <dl className="kv">
        <dt>Condition</dt>
        <dd>{alarm.key}</dd>
        <dt>Action taken</dt>
        <dd>{alarm.action || "—"}</dd>
        <dt>Occurrences</dt>
        <dd>{alarm.occurrence_count}</dd>
        <dt>Comment</dt>
        <dd>{alarm.ack_comment ?? "—"}</dd>
      </dl>
      <h3>Transitions</h3>
      <table>
        <thead>
          <tr>
            <th>At</th>
            <th>From</th>
            <th>To</th>
            <th>By</th>
            <th className="number">Value</th>
          </tr>
        </thead>
        <tbody>
          {transitions.map((transition) => (
            <tr key={transition.id}>
              <td>{formatDateTime(transition.at_ms)}</td>
              <td>{transition.from_state ? STATE_LABEL[transition.from_state] : "—"}</td>
              <td>{STATE_LABEL[transition.to_state]}</td>
              <td>{transition.actor}</td>
              <td className="number">{formatQuantity(transition.value, alarm.unit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ActiveAlarms({
  onSelect,
  selectedId,
}: {
  onSelect: (id: number) => void;
  selectedId: number | null;
}) {
  const active = useActiveAlarms();
  const acknowledge = useAcknowledge();
  const acknowledgeAll = useAcknowledgeAll();
  const mayAcknowledge = useCan("acknowledge_alarms");
  const alarms = sortAlarms(active.data ?? []);
  const unacknowledged = alarms.filter(isUnacknowledged).length;
  const failure = acknowledge.error ?? acknowledgeAll.error;

  return (
    <section className="panel" aria-label="Active alarms">
      <div className="row">
        <h2>Active alarms</h2>
        {mayAcknowledge && (
          <button
            type="button"
            disabled={unacknowledged === 0 || acknowledgeAll.isPending}
            onClick={() => {
              acknowledgeAll.mutate("");
            }}
          >
            Acknowledge all ({unacknowledged})
          </button>
        )}
      </div>
      {active.isError && <p className="error">{describeError(active.error)}</p>}
      {failure && (
        <p className="error" role="alert">
          {describeError(failure)}
        </p>
      )}
      {acknowledge.data && !acknowledge.data.accepted && (
        <p className="error" role="alert">
          Refused: {acknowledge.data.reason}
        </p>
      )}
      {alarms.length === 0 && !active.isLoading ? (
        <p className="muted">No active alarms.</p>
      ) : (
        <div className="table-scroll">
          <table>
            <AlarmTableHead withAction={mayAcknowledge} />
            <tbody>
              {alarms.map((alarm) => (
                <AlarmRow
                  key={alarm.id}
                  alarm={alarm}
                  selected={alarm.id === selectedId}
                  onSelect={() => {
                    onSelect(alarm.id);
                  }}
                  action={
                    mayAcknowledge ? (
                      isUnacknowledged(alarm) ? (
                        <button
                          type="button"
                          disabled={acknowledge.isPending}
                          onClick={() => {
                            acknowledge.mutate({ alarmId: alarm.id });
                          }}
                        >
                          Acknowledge
                        </button>
                      ) : (
                        <span className="muted">—</span>
                      )
                    ) : undefined
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function toMs(value: string): number | null {
  if (!value) {
    return null;
  }
  const ms = new Date(value).getTime();
  return Number.isFinite(ms) ? ms : null;
}

function AlarmHistory({
  onSelect,
  selectedId,
}: {
  onSelect: (id: number) => void;
  selectedId: number | null;
}) {
  const [draft, setDraft] = useState({ severity: "", parameter: "", from: "", to: "" });
  const [filter, setFilter] = useState<AlarmHistoryFilter>({ limit: HISTORY_PAGE, offset: 0 });
  const page = useQuery({
    queryKey: ["alarms", "history", filter],
    queryFn: ({ signal }) => fetchAlarmHistory(filter, signal),
  });

  const apply = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFilter({
      severity: draft.severity as AlarmHistoryFilter["severity"],
      parameter: draft.parameter.trim(),
      fromMs: toMs(draft.from),
      toMs: toMs(draft.to),
      limit: HISTORY_PAGE,
      offset: 0,
    });
  };

  const total = page.data?.total ?? 0;
  return (
    <section className="panel" aria-label="Alarm history">
      <h2>History</h2>
      <form className="form-grid" onSubmit={apply} aria-label="Alarm history filters">
        <label>
          Severity
          <select
            value={draft.severity}
            onChange={(event) => {
              setDraft({ ...draft, severity: event.target.value });
            }}
          >
            <option value="">any</option>
            <option value="warning">warning</option>
            <option value="critical">critical</option>
          </select>
        </label>
        <label>
          Parameter
          <input
            value={draft.parameter}
            placeholder="water_level_m"
            onChange={(event) => {
              setDraft({ ...draft, parameter: event.target.value });
            }}
          />
        </label>
        <label>
          Raised from
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
          Raised to
          <input
            type="datetime-local"
            step={1}
            value={draft.to}
            onChange={(event) => {
              setDraft({ ...draft, to: event.target.value });
            }}
          />
        </label>
        <button type="submit">Apply</button>
      </form>
      {page.isError && <p className="error">{describeError(page.error)}</p>}
      <div className="table-scroll">
        <table>
          <AlarmTableHead withAction={false} />
          <tbody>
            {(page.data?.items ?? []).map((alarm) => (
              <AlarmRow
                key={alarm.id}
                alarm={alarm}
                selected={alarm.id === selectedId}
                onSelect={() => {
                  onSelect(alarm.id);
                }}
              />
            ))}
          </tbody>
        </table>
      </div>
      <Pager
        offset={filter.offset}
        limit={filter.limit}
        total={total}
        onChange={(offset) => {
          setFilter({ ...filter, offset });
        }}
      />
    </section>
  );
}

export function AlarmsScreen() {
  const [tab, setTab] = useState<"active" | "history">("active");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  return (
    <div className="stack">
      <div className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "active"}
          onClick={() => {
            setTab("active");
          }}
        >
          Active
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "history"}
          onClick={() => {
            setTab("history");
          }}
        >
          History
        </button>
      </div>
      {tab === "active" ? (
        <ActiveAlarms onSelect={setSelectedId} selectedId={selectedId} />
      ) : (
        <AlarmHistory onSelect={setSelectedId} selectedId={selectedId} />
      )}
      {selectedId !== null && <AlarmDetailPanel alarmId={selectedId} />}
    </div>
  );
}
