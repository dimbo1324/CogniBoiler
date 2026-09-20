import { useQueryClient } from "@tanstack/react-query";
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

const LiveContext = createContext<LiveStore | null>(null);

/** Opens the live channels for the signed-in session and keeps them open. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const { manager } = useSession();
  const queryClient = useQueryClient();
  const [store] = useState(() => new LiveStore());

  useEffect(() => {
    const client = new RealtimeClient({
      url: realtimeUrl(window.location),
      channels: ["telemetry", "plc", "alarms"],
      maxRateHz: TELEMETRY_RATE_HZ,
      token: () => manager.accessToken(),
      handlers: {
        frame: (frame) => {
          store.apply(frame);
          if (frame.channel === "alarms") {
            void queryClient.invalidateQueries({ queryKey: queryKeys.alarms.all });
          }
          if (frame.channel === "plc" && frame.kind === "event") {
            void queryClient.invalidateQueries({ queryKey: queryKeys.plc.all });
          }
        },
        state: (state) => {
          store.setConnection(state);
        },
        unauthorized: () => manager.renew(),
        resync: () => {
          void queryClient.invalidateQueries();
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
