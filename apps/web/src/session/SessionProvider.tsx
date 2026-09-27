import { useQueryClient } from "@tanstack/react-query";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";

import type { Role } from "../api/types";
import { can, type Permission } from "./roles";
import { SessionManager, type SessionState, type SessionUser } from "./session";

interface SessionContextValue {
  manager: SessionManager;
  state: SessionState;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({
  children,
  manager: provided,
}: {
  children: ReactNode;
  manager?: SessionManager;
}) {
  const [manager] = useState(() => provided ?? new SessionManager());
  const state = useSyncExternalStore(manager.subscribe, manager.snapshot);
  const queryClient = useQueryClient();

  useEffect(() => {
    manager.attach();
    void manager.restore();
    return () => {
      manager.detach();
    };
  }, [manager]);

  // Cached answers belong to the user who fetched them. The refresh cookie is shared by all
  // tabs, so a renewal can switch this tab to another user without a sign-out in between:
  // clear on every change of identity, not only on the way out.
  const identity =
    state.status === "signed_in" ? `${state.user.username}:${state.user.role}` : null;
  const cachedFor = useRef<string | null>(null);
  useEffect(() => {
    if (cachedFor.current !== null && cachedFor.current !== identity) {
      queryClient.clear();
    }
    cachedFor.current = identity;
  }, [identity, queryClient]);

  const value = useMemo(() => ({ manager, state }), [manager, state]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (value === null) {
    throw new Error("useSession must be used inside SessionProvider");
  }
  return value;
}

/** The signed-in user; screens behind the sign-in gate may rely on it. */
export function useUser(): SessionUser {
  const { state } = useSession();
  if (state.status !== "signed_in") {
    throw new Error("useUser needs a signed-in session");
  }
  return state.user;
}

export function useRole(): Role | null {
  const { state } = useSession();
  return state.status === "signed_in" ? state.user.role : null;
}

export function useCan(permission: Permission): boolean {
  return can(useRole(), permission);
}
