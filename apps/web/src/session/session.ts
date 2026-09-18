// The signed-in session. The access token lives only in this object's memory; the refresh
// token only in the gateway's httpOnly cookie, which script cannot read. A reload therefore
// restores the session with one refresh, and closing the tab forgets the access token.

import { ApiError, bindCredentials, type Credentials } from "../api/http";
import { refreshSession, signIn, signOut } from "../api/endpoints";
import type { Role, TokenResponse } from "../api/types";

export interface SessionUser {
  username: string;
  role: Role;
}

export type SessionState =
  | { status: "restoring" }
  | { status: "signed_out"; notice: string | null }
  | { status: "signed_in"; user: SessionUser; sessionExpiresAtMs: number };

// Renew this long before the access token expires, so requests never meet an expired one.
const RENEW_AHEAD_MS = 60_000;
const MIN_RENEW_DELAY_MS = 5_000;
// Another tab exchanged the shared cookie a moment ago: the new cookie is already set.
const SUPERSEDED_RETRY_MS = 1_000;

const ROLES: readonly Role[] = ["viewer", "operator", "engineer", "admin"];

function asRole(value: string): Role {
  return (ROLES as readonly string[]).includes(value) ? (value as Role) : "viewer";
}

export function endedNotice(code: string): string {
  switch (code) {
    case "auth.session_invalid":
      return "Your session was closed or your account changed. Sign in again.";
    case "auth.refresh_reused":
      return "Your session was closed because its token was used twice. Sign in again.";
    default:
      return "Your session has ended. Sign in again.";
  }
}

export interface SessionTimers {
  setTimeout(callback: () => void, ms: number): ReturnType<typeof setTimeout>;
  clearTimeout(handle: ReturnType<typeof setTimeout>): void;
  now(): number;
}

const browserTimers: SessionTimers = {
  setTimeout: (callback, ms) => setTimeout(callback, ms),
  clearTimeout: (handle) => {
    clearTimeout(handle);
  },
  now: () => Date.now(),
};

export class SessionManager implements Credentials {
  private current: SessionState = { status: "restoring" };
  private token: string | null = null;
  private renewing: Promise<string | null> | null = null;
  private restoring: Promise<void> | null = null;
  private renewTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners = new Set<() => void>();
  private readonly tokenListeners = new Set<(token: string) => void>();
  private readonly timers: SessionTimers;

  constructor(timers: SessionTimers = browserTimers) {
    this.timers = timers;
  }

  // Credentials, for the HTTP layer

  accessToken(): string | null {
    return this.token;
  }

  renew(): Promise<string | null> {
    this.renewing ??= this.exchange().finally(() => {
      this.renewing = null;
    });
    return this.renewing;
  }

  ended(code: string): void {
    this.forget({ status: "signed_out", notice: endedNotice(code) });
  }

  // For React (useSyncExternalStore) and the live channels

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  readonly snapshot = (): SessionState => this.current;

  onTokenRenewed(listener: (token: string) => void): () => void {
    this.tokenListeners.add(listener);
    return () => {
      this.tokenListeners.delete(listener);
    };
  }

  attach(): void {
    bindCredentials(this);
  }

  detach(): void {
    bindCredentials(null);
    this.clearRenewTimer();
  }

  /**
   * At start-up: the refresh cookie, if the browser still has one, restores the session.
   * Called twice (React's StrictMode runs effects twice), it refreshes once: a second
   * exchange of the same cookie would be refused as a reuse.
   */
  restore(): Promise<void> {
    this.restoring ??= this.restoreOnce();
    return this.restoring;
  }

  private async restoreOnce(): Promise<void> {
    try {
      this.accept(await refreshSession());
    } catch (error) {
      const unreachable = error instanceof ApiError && error.code === "network.unreachable";
      this.forget({
        status: "signed_out",
        notice: unreachable
          ? "The gateway cannot be reached. Sign-in will work once it answers."
          : null,
      });
    }
  }

  async signIn(username: string, password: string): Promise<void> {
    this.accept(await signIn(username, password));
  }

  async signOut(): Promise<void> {
    try {
      await signOut();
    } finally {
      this.forget({ status: "signed_out", notice: null });
    }
  }

  private async exchange(): Promise<string | null> {
    try {
      return this.accept(await refreshSession());
    } catch (error) {
      if (error instanceof ApiError && error.code === "auth.refresh_superseded") {
        await new Promise<void>((resolve) => {
          this.timers.setTimeout(resolve, SUPERSEDED_RETRY_MS);
        });
        try {
          return this.accept(await refreshSession());
        } catch (again) {
          return this.fail(again);
        }
      }
      return this.fail(error);
    }
  }

  private fail(error: unknown): null {
    const code = error instanceof ApiError ? error.code : "network.unreachable";
    if (code === "network.unreachable" && this.token !== null) {
      // The gateway is down, not the session: keep it and try again on the next request.
      return null;
    }
    this.forget({ status: "signed_out", notice: endedNotice(code) });
    return null;
  }

  private accept(tokens: TokenResponse): string {
    this.token = tokens.access_token;
    this.current = {
      status: "signed_in",
      user: { username: tokens.username, role: asRole(tokens.role) },
      sessionExpiresAtMs: tokens.session_expires_at_ms,
    };
    this.scheduleRenewal(tokens.access_expires_at_ms);
    this.emit();
    for (const listener of this.tokenListeners) {
      listener(tokens.access_token);
    }
    return tokens.access_token;
  }

  private forget(state: SessionState): void {
    this.token = null;
    this.clearRenewTimer();
    this.current = state;
    this.emit();
  }

  private scheduleRenewal(accessExpiresAtMs: number): void {
    this.clearRenewTimer();
    const delay = Math.max(
      accessExpiresAtMs - this.timers.now() - RENEW_AHEAD_MS,
      MIN_RENEW_DELAY_MS,
    );
    this.renewTimer = this.timers.setTimeout(() => {
      this.renewTimer = null;
      void this.renew();
    }, delay);
  }

  private clearRenewTimer(): void {
    if (this.renewTimer !== null) {
      this.timers.clearTimeout(this.renewTimer);
      this.renewTimer = null;
    }
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}
