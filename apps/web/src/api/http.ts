// The only module that performs HTTP. Components and hooks call the typed functions in
// endpoints.ts, which come here, so authentication, token renewal and error decoding
// have exactly one home.

export interface FieldProblem {
  location: string;
  message: string;
}

export interface Problem {
  status: number;
  code: string;
  title: string;
  detail: string;
  errors: FieldProblem[];
  retryAfterS: number | null;
}

export class ApiError extends Error {
  readonly problem: Problem;

  constructor(problem: Problem) {
    super(problem.detail);
    this.name = "ApiError";
    this.problem = problem;
  }

  get status(): number {
    return this.problem.status;
  }

  get code(): string {
    return this.problem.code;
  }
}

/** What the HTTP layer needs from the signed-in session. */
export interface Credentials {
  accessToken(): string | null;
  /** A fresh access token after the gateway refused the current one; null if none. */
  renew(): Promise<string | null>;
  /** The gateway ended the session: it was closed, the account blocked, or it expired. */
  ended(code: string): void;
}

// Refused access tokens that a refresh can replace. Any other 401 means the session
// itself is over.
const RENEWABLE = new Set(["auth.token_expired", "auth.token_invalid", "auth.token_missing"]);

let credentials: Credentials | null = null;

export function bindCredentials(value: Credentials | null): void {
  credentials = value;
}

export type Query = Record<string, string | number | boolean | null | undefined>;

export interface RequestOptions {
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
  /** Send the access token and renew it once when refused. Default true. */
  auth?: boolean;
}

export function buildPath(path: string, query?: Query): string {
  if (!query) {
    return path;
  }
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  }
  const encoded = params.toString();
  return encoded ? `${path}?${encoded}` : path;
}

async function send(
  method: string,
  path: string,
  options: RequestOptions,
  token: string | null,
): Promise<Response> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    return await fetch(buildPath(path, options.query), {
      method,
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
      credentials: "same-origin",
    });
  } catch (error) {
    if (options.signal?.aborted) {
      throw error;
    }
    throw new ApiError({
      status: 0,
      code: "network.unreachable",
      title: "Network error",
      detail: "The gateway cannot be reached.",
      errors: [],
      retryAfterS: null,
    });
  }
}

function retryAfter(response: Response): number | null {
  const header = response.headers.get("Retry-After");
  if (header === null) {
    return null;
  }
  const seconds = Number(header);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

export async function readProblem(response: Response): Promise<Problem> {
  let body: Record<string, unknown> = {};
  try {
    const parsed: unknown = await response.json();
    if (parsed !== null && typeof parsed === "object") {
      body = parsed as Record<string, unknown>;
    }
  } catch {
    body = {};
  }
  const text = (key: string, fallback: string): string =>
    typeof body[key] === "string" ? body[key] : fallback;
  const errors = Array.isArray(body.errors)
    ? body.errors.filter(
        (item): item is FieldProblem =>
          item !== null &&
          typeof item === "object" &&
          typeof (item as FieldProblem).location === "string" &&
          typeof (item as FieldProblem).message === "string",
      )
    : [];
  return {
    status: response.status,
    code: text("code", `http.${String(response.status)}`),
    title: text("title", response.statusText || "Error"),
    detail: text("detail", `The gateway answered HTTP ${String(response.status)}.`),
    errors,
    retryAfterS: retryAfter(response),
  };
}

export async function request<T>(
  method: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const auth = options.auth ?? true;
  const session = auth ? credentials : null;
  let response = await send(method, path, options, session?.accessToken() ?? null);

  if (session && response.status === 401) {
    const problem = await readProblem(response);
    if (!RENEWABLE.has(problem.code)) {
      session.ended(problem.code);
      throw new ApiError(problem);
    }
    const token = await session.renew();
    if (token === null) {
      throw new ApiError(problem);
    }
    response = await send(method, path, options, token);
    if (response.status === 401) {
      const again = await readProblem(response);
      session.ended(again.code);
      throw new ApiError(again);
    }
  }

  if (!response.ok) {
    throw new ApiError(await readProblem(response));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

/** A sentence for the operator; never the raw exception text. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const fields = error.problem.errors.map((item) => `${item.location}: ${item.message}`);
    return fields.length ? `${error.problem.detail} ${fields.join("; ")}` : error.problem.detail;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong.";
}
