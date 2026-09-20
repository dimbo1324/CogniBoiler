import { useQuery } from "@tanstack/react-query";
import { useState, type FormEvent, type ReactNode } from "react";

import {
  isUnacknowledged,
  sortAlarms,
  useAcknowledge,
  useAcknowledgeAll,
  useActiveAlarms,
} from "../alarms/queries";
import { isCritical, severityGlyph, severityTone, SEVERITIES } from "../alarms/severity";
import { fetchAlarm, fetchAlarmHistory, type AlarmHistoryFilter } from "../api/endpoints";
import type { Alarm, AlarmState } from "../api/types";
import { CommandResult } from "../components/CommandResult";
import { Pager } from "../components/Pager";
import { Icon } from "../components/ui/Icon";
import { EmptyNote, ErrorOf } from "../components/ui/Note";
import { AlarmsIcon, AuditIcon, InfoIcon } from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";
import { useCan } from "../session/SessionProvider";
import { formatDateTime, formatQuantity, localInputToMs, parameterLabel } from "../units";

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
    if (isCritical(alarm.severity)) {
      classes.push("flashing");
    }
  }
  return (
    <tr
      className={classes.join(" ")}
      aria-selected={selected}
      data-testid={`alarm-${String(alarm.id)}`}
    >
      <td>
        <span className="severity-cell">
          <Icon glyph={severityGlyph(alarm.severity)} tone={severityTone(alarm.severity)} />
          {alarm.severity}
        </span>
      </td>
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
    return <ErrorOf error={detail.error} />;
  }
  if (!detail.data) {
    return <EmptyNote>Loading…</EmptyNote>;
  }
  const { alarm, transitions } = detail.data;
  return (
    <Panel
      title={`Alarm ${String(alarm.id)}: ${parameterLabel(alarm.parameter)} (${alarm.severity})`}
      label="Alarm details"
      glyph={severityGlyph(alarm.severity)}
    >
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
    </Panel>
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
    <Panel
      title="Active alarms"
      glyph={AlarmsIcon}
      actions={
        mayAcknowledge ? (
          <button
            type="button"
            disabled={unacknowledged === 0 || acknowledgeAll.isPending}
            onClick={() => {
              acknowledgeAll.mutate("");
            }}
          >
            <Icon glyph={AlarmsIcon} tone="muted" />
            Acknowledge all ({unacknowledged})
          </button>
        ) : undefined
      }
    >
      {active.isError && <ErrorOf error={active.error} />}
      <CommandResult result={acknowledge.data ?? undefined} error={failure} />
      {alarms.length === 0 && !active.isLoading ? (
        <EmptyNote>No active alarms.</EmptyNote>
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
    </Panel>
  );
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
      fromMs: localInputToMs(draft.from),
      toMs: localInputToMs(draft.to),
      limit: HISTORY_PAGE,
      offset: 0,
    });
  };

  const total = page.data?.total ?? 0;
  return (
    <Panel title="History" label="Alarm history" glyph={AuditIcon}>
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
            {SEVERITIES.map((severity) => (
              <option key={severity} value={severity}>
                {severity}
              </option>
            ))}
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
        <button type="submit">
          <Icon glyph={InfoIcon} tone="muted" />
          Apply
        </button>
      </form>
      {page.isError && <ErrorOf error={page.error} />}
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
    </Panel>
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
