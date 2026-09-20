import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

import { Horn } from "../alarms/horn";
import { annunciating, isUnacknowledged, useActiveAlarms } from "../alarms/queries";
import { parameterLabel } from "../units";
import { Icon } from "./ui/Icon";
import { AlarmsIcon, CriticalIcon, SilenceIcon } from "./ui/icons";

function countLabel(count: number, noun: string): string {
  return `${String(count)} ${noun}${count > 1 ? "s" : ""}`;
}

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
      <Icon glyph={critical.length ? CriticalIcon : AlarmsIcon} />
      <span>
        {critical.length
          ? countLabel(critical.length, "unacknowledged critical alarm")
          : countLabel(unacknowledged, "unacknowledged alarm")}
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
          <Icon glyph={SilenceIcon} />
          Silence
        </button>
      )}
    </div>
  );
}
