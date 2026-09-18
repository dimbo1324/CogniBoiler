import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

import { Horn } from "../alarms/horn";
import { annunciating, isUnacknowledged, useActiveAlarms } from "../alarms/queries";
import { parameterLabel } from "../units";

/**
 * Above every screen: flashes and sounds while an unacknowledged critical alarm stands.
 * Silencing stops the sound for the alarms standing now; a new one sounds again.
 */
export function AlarmBanner() {
  const { data: alarms = [] } = useActiveAlarms();
  const [horn] = useState(() => new Horn());
  const [silenced, setSilenced] = useState<ReadonlySet<number>>(new Set());

  const critical = useMemo(() => annunciating(alarms), [alarms]);
  const unacknowledged = alarms.filter(isUnacknowledged).length;
  const audible = critical.some((alarm) => !silenced.has(alarm.id));

  useEffect(() => {
    if (audible) {
      horn.start();
    } else {
      horn.stop();
    }
  }, [audible, horn]);

  useEffect(
    () => () => {
      horn.stop();
    },
    [horn],
  );

  if (unacknowledged === 0) {
    return null;
  }
  const first = critical[0];
  return (
    <div
      className={`alarm-banner${critical.length ? " flashing" : ""}`}
      role="alert"
      aria-live="assertive"
    >
      <span>
        {critical.length
          ? `${String(critical.length)} unacknowledged critical alarm${critical.length > 1 ? "s" : ""}`
          : `${String(unacknowledged)} unacknowledged alarm${unacknowledged > 1 ? "s" : ""}`}
        {first ? ` — ${parameterLabel(first.parameter)}: ${first.message}` : ""}
      </span>
      <Link to="/alarms">Open alarms</Link>
      {audible && (
        <button
          type="button"
          onClick={() => {
            setSilenced(new Set(critical.map((alarm) => alarm.id)));
          }}
        >
          Silence
        </button>
      )}
    </div>
  );
}
