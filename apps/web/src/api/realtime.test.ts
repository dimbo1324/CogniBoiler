import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import { plantState } from "../test/fixtures";
import {
  CLOSE_BAD_REQUEST,
  CLOSE_TRY_AGAIN_LATER,
  CLOSE_UNAUTHORIZED,
  RealtimeClient,
  realtimeUrl,
  type ConnectionState,
  type RealtimeHandlers,
  type SocketLike,
  type UnauthorizedAnswer,
} from "./realtime";
import type { DataFrame } from "./types";

class FakeSocket implements SocketLike {
  readyState = 0;
  sent: Record<string, unknown>[] = [];
  closedWith: number | null = null;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  send(data: string): void {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }

  close(code?: number): void {
    this.closedWith = code ?? 1000;
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  receive(message: Record<string, unknown>): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(message) }));
  }

  serverClose(code: number): void {
    this.readyState = 3;
    this.onclose?.({ code } as CloseEvent);
  }

  goLive(): void {
    this.open();
    this.receive({ type: "welcome", user: "operator", role: "operator" });
    this.receive({ type: "subscribed", channels: [] });
  }
}

describe("RealtimeClient", () => {
  let sockets: FakeSocket[];
  let states: ConnectionState[];
  let frames: DataFrame[];
  let handlers: RealtimeHandlers & {
    unauthorized: Mock<() => Promise<UnauthorizedAnswer>>;
    resync: Mock<() => void>;
  };
  let token: string | null;
  let logged: Mock<(...args: unknown[]) => void>;

  function client(backoffMs: readonly number[] = [100, 200]): RealtimeClient {
    return new RealtimeClient({
      url: "ws://console/ws",
      channels: ["telemetry", "alarms"],
      maxRateHz: 2,
      token: () => token,
      handlers,
      createSocket: () => {
        const socket = new FakeSocket();
        sockets.push(socket);
        return socket;
      },
      backoffMs,
    });
  }

  function latest(): FakeSocket {
    const socket = sockets[sockets.length - 1];
    if (socket === undefined) {
      throw new Error("no socket was opened");
    }
    return socket;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    sockets = [];
    states = [];
    frames = [];
    token = "token-1";
    handlers = {
      frame: (frame) => {
        frames.push(frame);
      },
      state: (state) => {
        states.push(state);
      },
      unauthorized: vi.fn<() => Promise<UnauthorizedAnswer>>(() =>
        Promise.resolve({ token: "token-2" }),
      ),
      resync: vi.fn<() => void>(),
    };
    logged = vi.fn<(...args: unknown[]) => void>();
    vi.spyOn(console, "error").mockImplementation(logged);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("authenticates in the first frame, subscribes after the welcome and goes live", () => {
    const realtime = client();
    realtime.start();
    const socket = latest();
    socket.open();

    expect(socket.sent[0]).toEqual({ type: "auth", access_token: "token-1" });
    socket.receive({ type: "welcome", user: "operator", role: "operator" });
    expect(socket.sent[1]).toEqual({
      type: "subscribe",
      channels: ["telemetry", "alarms"],
      max_rate_hz: 2,
    });
    socket.receive({ type: "subscribed", channels: ["alarms", "telemetry"] });
    expect(states).toEqual(["connecting", "live"]);
    expect(handlers.resync).not.toHaveBeenCalled();

    socket.receive({
      type: "data",
      channel: "telemetry",
      kind: "state",
      ts_ms: 1,
      data: plantState(),
    });
    expect(frames).toHaveLength(1);
    realtime.stop();
  });

  it("ignores frames that are not data, and reports one that is not JSON", () => {
    const realtime = client();
    realtime.start();
    const socket = latest();
    socket.open();
    socket.onmessage?.(new MessageEvent("message", { data: "not json" }));
    socket.receive({ type: "pong" });
    socket.receive({ type: "renewed", token_expires_at_ms: 1 });
    socket.receive({ type: "data", channel: "plc" });
    expect(frames).toHaveLength(0);
    expect(logged).toHaveBeenCalledTimes(1);
    realtime.stop();
  });

  it("reports the gateway's error and closing frames", () => {
    const realtime = client();
    realtime.start();
    latest().open();
    latest().receive({ type: "error", code: "ws.unknown_message", detail: "unknown type" });
    latest().receive({ type: "closing", code: 1013, reason: "client too slow; reload state" });
    expect(logged).toHaveBeenCalledTimes(2);
    expect(logged.mock.calls[0]).toContain("ws.unknown_message");
    expect(logged.mock.calls[1]).toContain("client too slow; reload state");
    realtime.stop();
  });

  it("stops and reports when the gateway refuses a malformed frame", () => {
    const realtime = client();
    realtime.start();
    latest().open();
    latest().serverClose(CLOSE_BAD_REQUEST);
    expect(realtime.state).toBe("stopped");
    expect(logged).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(1);
  });

  it("renews the token when the server refuses it and connects again after the backoff", async () => {
    const realtime = client();
    realtime.start();
    latest().open();
    token = "token-2";
    latest().serverClose(CLOSE_UNAUTHORIZED);
    await vi.waitFor(() => {
      expect(handlers.unauthorized).toHaveBeenCalledTimes(1);
    });
    expect(sockets).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(sockets).toHaveLength(2);

    latest().open();
    expect(latest().sent[0]).toEqual({ type: "auth", access_token: "token-2" });
    realtime.stop();
  });

  it("backs off between repeated refusals of a renewed token", async () => {
    const realtime = client([100, 200, 400]);
    realtime.start();
    const delays: number[] = [];
    for (let refusal = 0; refusal < 3; refusal += 1) {
      const before = sockets.length;
      latest().open();
      latest().serverClose(CLOSE_UNAUTHORIZED);
      let waited = 0;
      while (sockets.length === before && waited < 1000) {
        await vi.advanceTimersByTimeAsync(50);
        waited += 50;
      }
      delays.push(waited);
    }
    expect(delays).toEqual([100, 200, 400]);
    expect(handlers.unauthorized).toHaveBeenCalledTimes(3);
    realtime.stop();
  });

  it("keeps trying while the session is kept but could not be renewed now", async () => {
    handlers.unauthorized.mockResolvedValueOnce("retry");
    const realtime = client();
    realtime.start();
    latest().open();
    latest().serverClose(CLOSE_UNAUTHORIZED);
    await vi.waitFor(() => {
      expect(handlers.unauthorized).toHaveBeenCalledTimes(1);
    });
    expect(realtime.state).toBe("reconnecting");
    await vi.advanceTimersByTimeAsync(100);
    expect(sockets).toHaveLength(2);
    expect(states).not.toContain("stopped");
    realtime.stop();
  });

  it("stops for good when the session has ended", async () => {
    handlers.unauthorized.mockResolvedValue("ended");
    const realtime = client();
    realtime.start();
    latest().open();
    latest().serverClose(CLOSE_UNAUTHORIZED);
    await vi.waitFor(() => {
      expect(realtime.state).toBe("stopped");
    });
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(1);
  });

  it("asks for a token before connecting when the session has none", async () => {
    token = null;
    handlers.unauthorized.mockImplementation(() => {
      token = "token-2";
      return Promise.resolve({ token: "token-2" });
    });
    const realtime = client();
    realtime.start();
    expect(sockets).toHaveLength(0);
    await vi.waitFor(() => {
      expect(handlers.unauthorized).toHaveBeenCalledTimes(1);
    });
    await vi.advanceTimersByTimeAsync(100);
    expect(sockets).toHaveLength(1);
    realtime.stop();
  });

  it("does not reconnect when stopped while a renewal is on its way", async () => {
    let answer: (value: UnauthorizedAnswer) => void = () => undefined;
    handlers.unauthorized.mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    const realtime = client();
    realtime.start();
    latest().open();
    latest().serverClose(CLOSE_UNAUTHORIZED);
    realtime.stop();
    answer({ token: "token-2" });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(sockets).toHaveLength(1);
    expect(realtime.state).toBe("stopped");
  });

  it("reloads state once when live again after falling behind", () => {
    const realtime = client();
    realtime.start();
    latest().goLive();
    latest().serverClose(CLOSE_TRY_AGAIN_LATER);

    expect(realtime.state).toBe("reconnecting");
    expect(handlers.resync).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
    latest().goLive();
    expect(handlers.resync).toHaveBeenCalledTimes(1);
    realtime.stop();
  });

  it("reloads nothing on each failed attempt, only once when live again", () => {
    const realtime = client();
    realtime.start();
    latest().goLive();
    for (let attempt = 0; attempt < 4; attempt += 1) {
      latest().serverClose(1006);
      vi.advanceTimersByTime(200);
    }
    expect(handlers.resync).not.toHaveBeenCalled();
    latest().goLive();
    expect(handlers.resync).toHaveBeenCalledTimes(1);
    realtime.stop();
  });

  it("backs off between failed attempts and resets after a live connection", () => {
    const realtime = client();
    realtime.start();
    latest().serverClose(1006);
    vi.advanceTimersByTime(99);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    latest().serverClose(1006);
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(3);

    latest().open();
    latest().receive({ type: "subscribed", channels: [] });
    latest().serverClose(1006);
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(4);
    realtime.stop();
  });

  it("ignores the close of a socket it already replaced", () => {
    const realtime = client();
    realtime.start();
    const first = latest();
    const closeFirst = first.onclose;
    first.serverClose(1006);
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
    closeFirst?.({ code: 1006 } as CloseEvent);
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(2);
    realtime.stop();
  });

  it("hands a renewed token to the open connection and closes cleanly on stop", () => {
    const realtime = client();
    realtime.start();
    const socket = latest();
    realtime.renew("token-0");
    expect(socket.sent).toHaveLength(0);
    socket.open();
    realtime.renew("token-3");
    expect(socket.sent[1]).toEqual({ type: "auth", access_token: "token-3" });

    realtime.stop();
    expect(socket.closedWith).toBe(1000);
    expect(realtime.state).toBe("stopped");
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(1);
  });
});

describe("realtimeUrl", () => {
  it("uses the page's origin with the matching WebSocket scheme", () => {
    expect(realtimeUrl({ protocol: "http:", host: "localhost:5173" })).toBe(
      "ws://localhost:5173/ws",
    );
    expect(realtimeUrl({ protocol: "https:", host: "console.example" })).toBe(
      "wss://console.example/ws",
    );
  });
});
