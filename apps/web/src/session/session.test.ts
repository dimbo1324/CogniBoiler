import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/http";
import { tokens } from "../test/fixtures";
import {
  SIGN_OUT_FAILED_NOTICE,
  SessionManager,
  UNREACHABLE_NOTICE,
  endedNotice,
  type SessionTimers,
} from "./session";

vi.mock("../api/endpoints", () => ({
  refreshSession: vi.fn(),
  signIn: vi.fn(),
  signOut: vi.fn(),
}));

const endpoints = await import("../api/endpoints");
const refreshSession = vi.mocked(endpoints.refreshSession);
const signIn = vi.mocked(endpoints.signIn);
const signOut = vi.mocked(endpoints.signOut);

function apiError(status: number, code: string): ApiError {
  return new ApiError({
    status,
    code,
    title: "Error",
    detail: "refused",
    errors: [],
    retryAfterS: null,
  });
}

class ManualTimers implements SessionTimers {
  scheduled: { callback: () => void; ms: number }[] = [];
  nowMs = 1_000_000_000;

  setTimeout(callback: () => void, ms: number): ReturnType<typeof setTimeout> {
    this.scheduled.push({ callback, ms });
    return this.scheduled.length as unknown as ReturnType<typeof setTimeout>;
  }

  clearTimeout(): void {
    this.scheduled = [];
  }

  now(): number {
    return this.nowMs;
  }
}

describe("SessionManager", () => {
  let timers: ManualTimers;
  let manager: SessionManager;

  beforeEach(() => {
    timers = new ManualTimers();
    manager = new SessionManager(timers);
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it("restores a session from the refresh cookie, once however often it is asked", async () => {
    refreshSession.mockResolvedValue(tokens());
    await Promise.all([manager.restore(), manager.restore()]);

    expect(refreshSession).toHaveBeenCalledTimes(1);
    expect(manager.snapshot()).toMatchObject({
      status: "signed_in",
      user: { username: "operator", role: "operator" },
    });
    expect(manager.accessToken()).toBe("access-1");
  });

  it("shows the sign-in form without a notice when there is no cookie", async () => {
    refreshSession.mockRejectedValue(apiError(401, "auth.refresh_missing"));
    await manager.restore();
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: null });
  });

  it("schedules the renewal a minute before the access token expires", async () => {
    signIn.mockResolvedValue(tokens({ access_expires_at_ms: timers.nowMs + 900_000 }));
    await manager.signIn("operator", "secret-password");
    expect(timers.scheduled.at(-1)?.ms).toBe(840_000);
  });

  it("runs one refresh for concurrent renewals and tells the live channels", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession.mockResolvedValue(tokens({ access_token: "access-2" }));
    const renewed = vi.fn();
    manager.onTokenRenewed(renewed);

    const [first, second] = await Promise.all([manager.renew(), manager.renew()]);
    expect(first).toBe("access-2");
    expect(second).toBe("access-2");
    expect(refreshSession).toHaveBeenCalledTimes(1);
    expect(renewed).toHaveBeenCalledWith("access-2");
  });

  it("retries once when another tab exchanged the shared cookie a moment earlier", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession
      .mockRejectedValueOnce(apiError(401, "auth.refresh_superseded"))
      .mockResolvedValueOnce(tokens({ access_token: "access-3" }));

    const renewal = manager.renew();
    await vi.waitFor(() => {
      expect(timers.scheduled.some((entry) => entry.ms === 1000)).toBe(true);
    });
    timers.scheduled.find((entry) => entry.ms === 1000)?.callback();
    await expect(renewal).resolves.toBe("access-3");
  });

  it("signs out with a notice when the refresh is refused", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession.mockRejectedValue(apiError(401, "auth.refresh_reused"));

    await expect(manager.renew()).resolves.toBeNull();
    expect(manager.snapshot()).toEqual({
      status: "signed_out",
      notice: endedNotice("auth.refresh_reused"),
    });
    expect(manager.accessToken()).toBeNull();
  });

  it("keeps the session while the gateway is unreachable", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession.mockRejectedValue(apiError(0, "network.unreachable"));

    await expect(manager.renew()).resolves.toBeNull();
    expect(manager.snapshot().status).toBe("signed_in");
  });

  it.each([502, 503, 504])(
    "keeps the session and tries again soon when the proxy answers %i",
    async (status) => {
      signIn.mockResolvedValue(tokens());
      await manager.signIn("operator", "secret-password");
      timers.scheduled = [];
      refreshSession.mockRejectedValueOnce(apiError(status, `http.${String(status)}`));

      await expect(manager.renew()).resolves.toBeNull();
      expect(manager.snapshot().status).toBe("signed_in");
      expect(manager.accessToken()).toBe("access-1");
      expect(timers.scheduled).toHaveLength(1);
      expect(timers.scheduled[0]?.ms).toBe(5_000);

      refreshSession.mockResolvedValueOnce(tokens({ access_token: "access-2" }));
      timers.scheduled[0]?.callback();
      await vi.waitFor(() => {
        expect(manager.accessToken()).toBe("access-2");
      });
    },
  );

  it.each([
    [401, "auth.refresh_invalid"],
    [403, "auth.forbidden"],
    [500, "internal"],
  ])("signs out when the refresh is refused with %i", async (status, code) => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession.mockRejectedValue(apiError(status, code));

    await expect(manager.renew()).resolves.toBeNull();
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: endedNotice(code) });
  });

  it("says the gateway cannot be reached when the proxy answers 503 at start-up", async () => {
    refreshSession.mockRejectedValue(apiError(503, "http.503"));
    await manager.restore();
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: UNREACHABLE_NOTICE });
  });

  it("renews when the renewal timer fires", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession.mockResolvedValue(tokens({ access_token: "access-2" }));

    timers.scheduled.at(-1)?.callback();
    await vi.waitFor(() => {
      expect(manager.accessToken()).toBe("access-2");
    });
    expect(refreshSession).toHaveBeenCalledTimes(1);
  });

  it("stops the renewal timer when detached", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    expect(timers.scheduled).toHaveLength(1);
    manager.detach();
    expect(timers.scheduled).toHaveLength(0);
  });

  it("signs out when the retry after a superseded refresh is refused too", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    refreshSession
      .mockRejectedValueOnce(apiError(401, "auth.refresh_superseded"))
      .mockRejectedValueOnce(apiError(401, "auth.refresh_superseded"));

    const renewal = manager.renew();
    await vi.waitFor(() => {
      expect(timers.scheduled.some((entry) => entry.ms === 1000)).toBe(true);
    });
    timers.scheduled.find((entry) => entry.ms === 1000)?.callback();
    await expect(renewal).resolves.toBeNull();
    expect(refreshSession).toHaveBeenCalledTimes(2);
    expect(manager.snapshot().status).toBe("signed_out");
  });

  it("forgets the token on sign-out and says so when the gateway does not answer", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    signOut.mockRejectedValue(apiError(502, "http.502"));

    await expect(manager.signOut()).resolves.toBeUndefined();
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: SIGN_OUT_FAILED_NOTICE });
    expect(manager.accessToken()).toBeNull();
  });

  it("signs out with the access token, so the gateway closes the session without the cookie", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    signOut.mockResolvedValue({ message: "Signed out." });

    await manager.signOut();
    expect(signOut).toHaveBeenCalledWith("access-1");
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: null });
  });

  it("stays signed out when a renewal in flight answers after the sign-out", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    let answer: (value: ReturnType<typeof tokens>) => void = () => undefined;
    refreshSession.mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    signOut.mockResolvedValue({ message: "Signed out." });

    const renewal = manager.renew();
    await manager.signOut();
    answer(tokens({ access_token: "access-2" }));
    await expect(renewal).resolves.toBeNull();
    expect(manager.snapshot()).toEqual({ status: "signed_out", notice: null });
    expect(manager.accessToken()).toBeNull();
    expect(timers.scheduled).toHaveLength(0);
  });

  it("ends the session with the reason the gateway gave", async () => {
    signIn.mockResolvedValue(tokens());
    await manager.signIn("operator", "secret-password");
    const listener = vi.fn();
    manager.subscribe(listener);

    manager.ended("auth.session_invalid");
    expect(manager.snapshot()).toEqual({
      status: "signed_out",
      notice: endedNotice("auth.session_invalid"),
    });
    expect(listener).toHaveBeenCalled();
  });
});
