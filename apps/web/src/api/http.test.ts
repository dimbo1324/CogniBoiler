import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { jsonResponse, problem } from "../test/fixtures";
import {
  ApiError,
  bindCredentials,
  buildPath,
  describeError,
  request,
  type Credentials,
} from "./http";

function credentials(overrides: Partial<Credentials> = {}): Credentials {
  return {
    accessToken: () => "old-token",
    renew: vi.fn(() => Promise.resolve("new-token")),
    ended: vi.fn(),
    ...overrides,
  };
}

function authorization(call: unknown[]): string | undefined {
  const init = call[1] as RequestInit;
  return (init.headers as Record<string, string>).Authorization;
}

describe("request", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    bindCredentials(null);
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("sends the access token and decodes the JSON body", async () => {
    bindCredentials(credentials());
    fetchMock.mockResolvedValue(jsonResponse(200, { mode: "auto" }));

    await expect(request("GET", "/api/v1/plc/status")).resolves.toEqual({ mode: "auto" });
    expect(authorization(fetchMock.mock.calls[0] ?? [])).toBe("Bearer old-token");
  });

  it("renews an expired access token once and repeats the request", async () => {
    const session = credentials();
    bindCredentials(session);
    fetchMock
      .mockResolvedValueOnce(problem(401, "auth.token_expired"))
      .mockResolvedValueOnce(jsonResponse(200, { ok: true }));

    await expect(request("GET", "/api/v1/plant")).resolves.toEqual({ ok: true });
    expect(session.renew).toHaveBeenCalledTimes(1);
    expect(authorization(fetchMock.mock.calls[1] ?? [])).toBe("Bearer new-token");
    expect(session.ended).not.toHaveBeenCalled();
  });

  it("ends the session when the gateway closed it, without trying to renew", async () => {
    const session = credentials();
    bindCredentials(session);
    fetchMock.mockResolvedValue(problem(401, "auth.session_invalid"));

    await expect(request("GET", "/api/v1/plant")).rejects.toBeInstanceOf(ApiError);
    expect(session.renew).not.toHaveBeenCalled();
    expect(session.ended).toHaveBeenCalledWith("auth.session_invalid");
  });

  it("sends no token and never renews for a public route", async () => {
    const session = credentials();
    bindCredentials(session);
    fetchMock.mockResolvedValue(
      problem(401, "auth.invalid_credentials", "Invalid username or password."),
    );

    const failure = request("POST", "/auth/login", {
      body: { username: "a", password: "b" },
      auth: false,
    });
    await expect(failure).rejects.toMatchObject({ code: "auth.invalid_credentials", status: 401 });
    expect(authorization(fetchMock.mock.calls[0] ?? [])).toBeUndefined();
    expect(session.renew).not.toHaveBeenCalled();
  });

  it("reads Problem Details, validation errors and Retry-After", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        429,
        {
          code: "auth.too_many_attempts",
          title: "Too Many Requests",
          detail: "Too many failed sign-in attempts.",
          errors: [{ location: "body.username", message: "too short" }],
        },
        { "Retry-After": "600" },
      ),
    );

    const error = await request("POST", "/auth/login", { auth: false }).catch(
      (caught: unknown) => caught,
    );
    expect(error).toBeInstanceOf(ApiError);
    const problemDetails = (error as ApiError).problem;
    expect(problemDetails.retryAfterS).toBe(600);
    expect(problemDetails.errors).toEqual([{ location: "body.username", message: "too short" }]);
    expect(describeError(error)).toBe("Too many failed sign-in attempts. body.username: too short");
  });

  it("turns a network failure into a stable error code", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(request("GET", "/health", { auth: false })).rejects.toMatchObject({
      code: "network.unreachable",
      status: 0,
    });
  });

  it("gives an answer that is not JSON a generic problem", async () => {
    fetchMock.mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 }));

    await expect(request("GET", "/api/v1/plant", { auth: false })).rejects.toMatchObject({
      code: "http.502",
      status: 502,
    });
  });
});

describe("buildPath", () => {
  it("drops empty values and encodes the rest", () => {
    expect(
      buildPath("/api/v1/audit", { username: "a b", method: "", from_ms: null, limit: 10 }),
    ).toBe("/api/v1/audit?username=a+b&limit=10");
    expect(buildPath("/api/v1/plant")).toBe("/api/v1/plant");
  });
});
