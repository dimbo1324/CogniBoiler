import { useQueryClient, type QueryKey } from "@tanstack/react-query";
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";

import { RealtimeClient, realtimeUrl } from "../api/realtime";
import { useSession } from "../session/SessionProvider";
import { LiveStore, type LiveSnapshot } from "./store";
import { queryKeys } from "../api/queryKeys";

// Twice a second is smooth on the mimic and trends and light on a laptop; the gateway caps
// every client at 10 Hz anyway.
const TELEMETRY_RATE_HZ = 2;
// A trip raises several alarms and PLC events at once: one reload for the whole burst.
const INVALIDATE_COALESCE_MS = 250;
// What the stream keeps current; history ranges and other REST answers do not change with it.
const STREAM_KEYS: readonly QueryKey[] = [queryKeys.alarms.all, queryKeys.plc.all];

const LiveContext = createContext<LiveStore | null>(null);

/** Opens the live channels for the signed-in session and keeps them open. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const { manager } = useSession();
  const queryClient = useQueryClient();
  const [store] = useState(() => new LiveStore());

  useEffect(() => {
    const stale = new Map<string, QueryKey>();
    let flush: ReturnType<typeof setTimeout> | null = null;
    const invalidateSoon = (queryKey: QueryKey) => {
      stale.set(JSON.stringify(queryKey), queryKey);
      flush ??= setTimeout(() => {
        flush = null;
        for (const key of stale.values()) {
          void queryClient.invalidateQueries({ queryKey: key });
        }
        stale.clear();
      }, INVALIDATE_COALESCE_MS);
    };
    const client = new RealtimeClient({
      url: realtimeUrl(window.location),
      channels: ["telemetry", "plc", "alarms"],
      maxRateHz: TELEMETRY_RATE_HZ,
      token: () => manager.accessToken(),
      handlers: {
        frame: (frame) => {
          store.apply(frame);
          if (frame.channel === "alarms") {
            invalidateSoon(queryKeys.alarms.all);
          }
          if (frame.channel === "plc" && frame.kind === "event") {
            invalidateSoon(queryKeys.plc.all);
          }
        },
        state: (state) => {
          store.setConnection(state);
        },
        unauthorized: async () => {
          const token = await manager.renew();
          if (token !== null) {
            return { token };
          }
          return manager.snapshot().status === "signed_in" ? "retry" : "ended";
        },
        resync: () => {
          for (const queryKey of STREAM_KEYS) {
            void queryClient.invalidateQueries({ queryKey });
          }
        },
      },
    });
    const stopRenewals = manager.onTokenRenewed((token) => {
      client.renew(token);
    });
    client.start();
    return () => {
      stopRenewals();
      client.stop();
      if (flush !== null) {
        clearTimeout(flush);
      }
    };
  }, [manager, queryClient, store]);

  return <LiveContext.Provider value={store}>{children}</LiveContext.Provider>;
}

export function useLiveStore(): LiveStore {
  const store = useContext(LiveContext);
  if (store === null) {
    throw new Error("useLive must be used inside LiveProvider");
  }
  return store;
}

export function useLive(): LiveSnapshot {
  const store = useLiveStore();
  return useSyncExternalStore(store.subscribe, store.snapshot);
}
