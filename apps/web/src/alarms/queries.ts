import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { acknowledgeAlarm, acknowledgeAllAlarms, fetchActiveAlarms } from "../api/endpoints";
import type { Alarm } from "../api/types";
import { isCritical } from "./severity";
import { queryKeys } from "../api/queryKeys";

// Alarm changes arrive on the WebSocket and invalidate this query at once; the interval
// only covers a connection that is down.
const ACTIVE_ALARMS_REFRESH_MS = 15_000;

export function useActiveAlarms() {
  return useQuery({
    queryKey: queryKeys.alarms.active,
    queryFn: ({ signal }) => fetchActiveAlarms(signal),
    refetchInterval: ACTIVE_ALARMS_REFRESH_MS,
  });
}

export function useAcknowledge() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ alarmId, comment }: { alarmId: number; comment?: string }) =>
      acknowledgeAlarm(alarmId, comment),
    onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.alarms.all }),
  });
}

export function useAcknowledgeAll() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (comment?: string) => acknowledgeAllAlarms(comment),
    onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.alarms.all }),
  });
}

export function isUnacknowledged(alarm: Pick<Alarm, "state">): boolean {
  return alarm.state === "ACTIVE_UNACK" || alarm.state === "CLEARED_UNACK";
}

/** Unacknowledged critical alarms: these flash and sound until someone acknowledges them. */
export function annunciating(alarms: readonly Alarm[]): Alarm[] {
  return alarms.filter((alarm) => isCritical(alarm.severity) && isUnacknowledged(alarm));
}

/** Critical before warning, unacknowledged before acknowledged, newest first. */
export function sortAlarms(alarms: readonly Alarm[]): Alarm[] {
  const rank = (alarm: Alarm) =>
    (isCritical(alarm.severity) ? 0 : 2) + (isUnacknowledged(alarm) ? 0 : 1);
  return [...alarms].sort((a, b) => rank(a) - rank(b) || b.raised_at_ms - a.raised_at_ms);
}
