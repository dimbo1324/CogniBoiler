// The live channels of the gateway's WebSocket /ws. Framework-free, so it can be tested with
// a fake socket; LiveProvider owns one instance per signed-in session.
//
// Protocol: the access token goes in the first frame (never in the URL, so it stays out of
// access logs); after "welcome" the client subscribes. The server closes with 4401 when the
// token expires or the session ends, 4400 on a malformed frame and 1013 when this client
// fell behind on events — then state must be reloaded over REST before trusting the stream.

import type { Channel, DataFrame } from "./types";

export const CLOSE_UNAUTHORIZED = 4401;
export const CLOSE_BAD_REQUEST = 4400;
export const CLOSE_TRY_AGAIN_LATER = 1013;

export type ConnectionState = "connecting" | "live" | "reconnecting" | "stopped";

export interface RealtimeHandlers {
  frame(frame: DataFrame): void;
  state(state: ConnectionState): void;
  /** The server refused the token: renew it; null ends the connection for good. */
  unauthorized(): Promise<string | null>;
  /** Frames may have been lost; reload whatever the stream keeps current. */
  resync(): void;
}

export interface SocketLike {
  readonly readyState: number;
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

export interface RealtimeOptions {
  url: string;
  channels: readonly Channel[];
  maxRateHz: number;
  token: () => string | null;
  handlers: RealtimeHandlers;
  createSocket?: (url: string) => SocketLike;
  /** Delays before successive reconnect attempts [ms]; the last one repeats. */
  backoffMs?: readonly number[];
}

const OPEN = 1;
const DEFAULT_BACKOFF_MS = [500, 1000, 2000, 5000, 10000] as const;

export function realtimeUrl(location: Pick<Location, "protocol" | "host">): string {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws`;
}

function isDataFrame(
  message: Record<string, unknown>,
): message is DataFrame & Record<string, unknown> {
  return (
    message.type === "data" &&
    typeof message.channel === "string" &&
    typeof message.kind === "string" &&
    message.data !== null &&
    typeof message.data === "object"
  );
}

export class RealtimeClient {
  private readonly options: RealtimeOptions;
  private readonly backoff: readonly number[];
  private socket: SocketLike | null = null;
  private attempt = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private connectionState: ConnectionState = "stopped";

  constructor(options: RealtimeOptions) {
    this.options = options;
    this.backoff = options.backoffMs ?? DEFAULT_BACKOFF_MS;
  }

  get state(): ConnectionState {
    return this.connectionState;
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    this.connect("connecting");
  }

  stop(): void {
    this.running = false;
    this.clearTimer();
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onclose = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onopen = null;
      socket.close(1000, "client stopped");
    }
    this.setState("stopped");
  }

  /** Hand a renewed access token to the open connection, so the server does not close it. */
  renew(token: string): void {
    if (this.socket?.readyState === OPEN) {
      this.socket.send(JSON.stringify({ type: "auth", access_token: token }));
    }
  }

  private connect(state: ConnectionState): void {
    this.clearTimer();
    const token = this.options.token();
    if (!this.running) {
      return;
    }
    if (token === null) {
      void this.handleUnauthorized();
      return;
    }
    this.setState(state);
    const create = this.options.createSocket ?? ((url: string) => new WebSocket(url));
    const socket = create(this.options.url);
    this.socket = socket;
    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "auth", access_token: token }));
    };
    socket.onmessage = (event: MessageEvent) => {
      this.receive(socket, event.data);
    };
    socket.onclose = (event: CloseEvent) => {
      if (this.socket === socket) {
        this.socket = null;
        this.closed(event.code);
      }
    };
    socket.onerror = () => {
      // A close event always follows; reconnecting is decided there.
    };
  }

  private receive(socket: SocketLike, raw: unknown): void {
    if (typeof raw !== "string") {
      return;
    }
    let message: unknown;
    try {
      message = JSON.parse(raw);
    } catch {
      return;
    }
    if (message === null || typeof message !== "object") {
      return;
    }
    const frame = message as Record<string, unknown>;
    if (frame.type === "welcome") {
      socket.send(
        JSON.stringify({
          type: "subscribe",
          channels: this.options.channels,
          max_rate_hz: this.options.maxRateHz,
        }),
      );
      return;
    }
    if (frame.type === "subscribed") {
      this.attempt = 0;
      this.setState("live");
      return;
    }
    if (isDataFrame(frame)) {
      this.options.handlers.frame(frame);
    }
  }

  private closed(code: number): void {
    if (!this.running) {
      return;
    }
    if (code === CLOSE_UNAUTHORIZED) {
      void this.handleUnauthorized();
      return;
    }
    if (code === CLOSE_TRY_AGAIN_LATER) {
      this.options.handlers.resync();
    }
    this.scheduleReconnect();
  }

  private async handleUnauthorized(): Promise<void> {
    this.setState("reconnecting");
    const token = await this.options.handlers.unauthorized();
    if (!this.running) {
      return;
    }
    if (token === null) {
      this.stop();
      return;
    }
    this.connect("reconnecting");
  }

  private scheduleReconnect(): void {
    this.setState("reconnecting");
    const delay = this.backoff[Math.min(this.attempt, this.backoff.length - 1)] ?? 1000;
    this.attempt += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.options.handlers.resync();
      this.connect("reconnecting");
    }, delay);
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private setState(state: ConnectionState): void {
    if (state !== this.connectionState) {
      this.connectionState = state;
      this.options.handlers.state(state);
    }
  }
}
