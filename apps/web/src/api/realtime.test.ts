import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import { plantState } from "../test/fixtures";
import {
  CLOSE_TRY_AGAIN_LATER,
  CLOSE_UNAUTHORIZED,
  RealtimeClient,
  realtimeUrl,
  type ConnectionState,
  type RealtimeHandlers,
  type SocketLike,
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
}

describe("RealtimeClient", () => {
  let sockets: FakeSocket[];
  let states: ConnectionState[];
  let frames: DataFrame[];
  let handlers: RealtimeHandlers & {
    unauthorized: Mock<() => Promise<string | null>>;
    resync: Mock<() => void>;
  };
  let token: string | null;

  function client(): RealtimeClient {
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
      backoffMs: [100, 200],
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
      unauthorized: vi.fn<() => Promise<string | null>>(() => Promise.resolve("token-2")),
      resync: vi.fn<() => void>(),
    };
  });

  afterEach(() => {
    vi.useRealTimers();
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

  it("ignores frames that are not data or not JSON", () => {
    const realtime = client();
    realtime.start();
    const socket = latest();
    socket.open();
    socket.onmessage?.(new MessageEvent("message", { data: "not json" }));
    socket.receive({ type: "pong" });
    socket.receive({ type: "data", channel: "plc" });
    expect(frames).toHaveLength(0);
    realtime.stop();
  });

  it("renews the token when the server refuses it and connects again", async () => {
    const realtime = client();
    realtime.start();
    latest().open();
    token = "token-2";
    latest().serverClose(CLOSE_UNAUTHORIZED);
    await vi.waitFor(() => {
      expect(sockets).toHaveLength(2);
    });

    expect(handlers.unauthorized).toHaveBeenCalledTimes(1);
    latest().open();
    expect(latest().sent[0]).toEqual({ type: "auth", access_token: "token-2" });
    realtime.stop();
  });

  it("stops for good when the session cannot be renewed", async () => {
    handlers.unauthorized.mockResolvedValue(null);
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

  it("reloads state and reconnects after falling behind", () => {
    const realtime = client();
    realtime.start();
    latest().open();
    latest().serverClose(CLOSE_TRY_AGAIN_LATER);

    expect(handlers.resync).toHaveBeenCalled();
    expect(realtime.state).toBe("reconnecting");
    vi.advanceTimersByTime(100);
    expect(sockets).toHaveLength(2);
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

  it("hands a renewed token to the open connection and closes cleanly on stop", () => {
    const realtime = client();
    realtime.start();
    const socket = latest();
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
