// The README's pictures, taken from the running stack: the main screens of the console, and
// the frames of a GIF that shows the five-minute demo from the operator's overview.
//
// This records; it checks nothing the e2e suite does not check already. The actions that
// happen off the operator's screen — the engineer's fault, the acknowledgement, the reset —
// go through the gateway's API, so the recorded page stays on the overview the whole time.
// `python dev_tools_scripts_runner.py readme-media` runs this and turns the raw captures in
// README_MEDIA_DIR into the optimized files of docs/images/.

import { expect, request, test, type APIRequestContext, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { demoPassword, signIn, type DemoRole } from "../e2e/support";

const OUT = resolve(process.env.README_MEDIA_DIR ?? "readme-results/raw");
const SPEED = 10;
// The restart runs at three times real speed. At ten, the PLC's commands reach the plant a
// step late and a restart can trip again on high steam temperature: known defect Д17 of
// the roadmap. A recording must show what the unit does, not what that defect does.
const RESTART_SPEED = 3;
const WATTS_PER_MEGAWATT = 1e6;
const NOMINAL_MW = 250;
const TARGET_MW = 300;
const SCENARIO = "steady_state";
const FAULT = { kind: "feedwater_pump_failure", target: "", severity: 1, ramp_s: 0 };
// How often the overview is captured while something is happening, and how long the GIF
// holds a moment worth reading: the trip, the reset, the unit back on load.
const FRAME_EVERY_MS = 1_500;
const SLOW_FRAME_EVERY_MS = 4_000;
const KEY_HOLD_MS = 2_000;

/** The gateway as one demo user sees it, for the steps that happen off screen. */
class Api {
  private constructor(
    private readonly context: APIRequestContext,
    private readonly token: string,
  ) {}

  static async as(baseURL: string | undefined, role: DemoRole): Promise<Api> {
    const context = await request.newContext({ baseURL, ignoreHTTPSErrors: true });
    const login = await context.post("/auth/login", {
      data: { username: role, password: demoPassword(role) },
    });
    expect(login.ok(), `${role} signs in to the gateway`).toBe(true);
    const { access_token: token } = (await login.json()) as { access_token: string };
    return new Api(context, token);
  }

  private get headers() {
    return { Authorization: `Bearer ${this.token}` };
  }

  /** A POST the gateway must accept: a PLC refusal is a 200 with `accepted: false`. */
  async post(path: string, data: unknown = {}): Promise<void> {
    const reply = await this.context.post(path, { data, headers: this.headers });
    expect(reply.ok(), `POST ${path}`).toBe(true);
    const body = (await reply.json()) as { accepted?: boolean; reason?: string };
    expect(body.accepted ?? true, `POST ${path}: ${body.reason ?? ""}`).toBe(true);
  }

  async remove(path: string): Promise<void> {
    const reply = await this.context.delete(path, { headers: this.headers });
    expect(reply.ok(), `DELETE ${path}`).toBe(true);
  }

  async get<T>(path: string): Promise<T> {
    const reply = await this.context.get(path, { headers: this.headers });
    expect(reply.ok(), `GET ${path}`).toBe(true);
    return (await reply.json()) as T;
  }

  async close(): Promise<void> {
    await this.context.dispose();
  }
}

interface PlcStatus {
  emergency_stop_active: boolean;
  reset_permitted: boolean;
}

interface Frame {
  file: string;
  at_ms: number;
  caption: string;
  hold_ms: number;
  key: boolean;
}

/** The overview, captured frame by frame, with a manifest the GIF is assembled from. */
class Frames {
  private readonly frames: Frame[] = [];
  private readonly started = Date.now();

  constructor(private readonly page: Page) {}

  async shot(caption: string, key = false): Promise<void> {
    const file = `frame-${String(this.frames.length).padStart(4, "0")}.png`;
    await this.page.screenshot({ path: join(OUT, file) });
    this.frames.push({
      file,
      at_ms: Date.now() - this.started,
      caption,
      hold_ms: key ? KEY_HOLD_MS : 0,
      key,
    });
  }

  /** Capture every `everyMs` until `done` holds; the last frame shows it holding. */
  async until(
    caption: string,
    done: () => Promise<boolean>,
    timeoutMs: number,
    everyMs = FRAME_EVERY_MS,
  ): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!(await done())) {
      if (Date.now() > deadline) {
        throw new Error(`timed out waiting for: ${caption}`);
      }
      await this.shot(caption);
      await this.page.waitForTimeout(everyMs);
    }
  }

  write(): void {
    writeFileSync(join(OUT, "frames.json"), JSON.stringify(this.frames, null, 2) + "\n");
  }
}

function nav(page: Page, name: string) {
  return page.getByRole("navigation", { name: "Screens" }).getByRole("link", { name, exact: true });
}

async function outputMw(page: Page): Promise<number> {
  const text = (await page.getByTestId("mimic-power").textContent()) ?? "";
  return Number(/(-?\d+(?:\.\d+)?)/u.exec(text)?.[1] ?? Number.NaN);
}

async function plcMode(page: Page): Promise<string> {
  return (await page.getByRole("banner").getByLabel("PLC mode").textContent()) ?? "";
}

async function picture(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: join(OUT, `${name}.png`) });
}

/** The unit at nominal, no fault and no latched trip, in real time — what the demo leaves. */
async function nominal(engineer: Api, speed: number): Promise<void> {
  await engineer.remove("/api/v1/simulation/faults");
  // A latched trip is reset before the scenario is reloaded, never after: reloading a hot
  // plant behind shut valves sends the superheater to furnace temperature within a step
  // and trips the unit again (known defect Д17). The reset waits for the PLC to allow it.
  await expect
    .poll(
      async () => {
        const plc = await engineer.get<PlcStatus>("/api/v1/plc/status");
        return !plc.emergency_stop_active || plc.reset_permitted;
      },
      { timeout: 180_000 },
    )
    .toBe(true);
  if ((await engineer.get<PlcStatus>("/api/v1/plc/status")).emergency_stop_active) {
    await engineer.post("/api/v1/commands/reset", { operator_id: "readme" });
  }
  await engineer.post("/api/v1/simulation/scenario", { name: SCENARIO });
  await engineer.post("/api/v1/commands/load", { load_w: NOMINAL_MW * WATTS_PER_MEGAWATT });
  await engineer.post("/api/v1/simulation/resume");
  await engineer.post("/api/v1/simulation/speed", { speed_factor: speed });
}

test("the README's screenshots and the frames of its demo GIF", async ({ browser, baseURL }) => {
  mkdirSync(OUT, { recursive: true });
  const engineerApi = await Api.as(baseURL, "engineer");
  const operatorApi = await Api.as(baseURL, "operator");
  const operator = await (await browser.newContext()).newPage();
  const engineer = await (await browser.newContext()).newPage();
  try {
    await nominal(engineerApi, SPEED);
    await operatorApi.post("/api/v1/alarms/ack-all");
    await signIn(operator, "operator");
    await signIn(engineer, "engineer");

    // The unit at nominal, then on its way to 300 MW: the screens at their most ordinary.
    await expect.poll(() => outputMw(operator), { timeout: 120_000 }).toBeGreaterThan(240);
    await operatorApi.post("/api/v1/commands/load", {
      load_w: TARGET_MW * WATTS_PER_MEGAWATT,
    });
    await expect.poll(() => outputMw(operator), { timeout: 240_000 }).toBeGreaterThan(295);
    await picture(operator, "overview");

    await nav(operator, "Trends").click();
    await expect(operator.getByTestId("trend-electrical_power")).toBeVisible();
    // Long enough for the live series to draw a line rather than a dot.
    await operator.waitForTimeout(8_000);
    await picture(operator, "trends");

    await nav(operator, "Control").click();
    await expect(operator.getByRole("region", { name: "Load" })).toBeVisible();
    await picture(operator, "control");
    await nav(operator, "Overview").click();
    await expect(operator.getByTestId("mimic-power")).toBeVisible();

    // The GIF: the pump fails, the drum empties, the interlock trips the unit, the operator
    // acknowledges, the engineer repairs and resets, and the unit comes back on load.
    const frames = new Frames(operator);
    await frames.shot(`${String(TARGET_MW)} MW in AUTO`, true);
    await engineerApi.post("/api/v1/simulation/faults", FAULT);
    await frames.until(
      "feedwater pump failed: the drum level falls",
      async () => (await plcMode(operator)) === "E-STOP",
      240_000,
    );
    await expect(operator.getByRole("alert").filter({ hasText: "critical" })).toBeVisible();
    await frames.shot("the interlock trips the unit; a critical alarm flashes", true);
    await picture(operator, "trip");

    await nav(engineer, "Engineer").click();
    await expect(engineer.getByTestId("fault-feedwater_pump_failure")).toBeVisible();
    await picture(engineer, "engineer");
    await nav(engineer, "Alarms").click();
    await expect(engineer.getByRole("region", { name: "Active alarms" })).toBeVisible();
    await picture(engineer, "alarms");

    await operatorApi.post("/api/v1/alarms/ack-all");
    await frames.shot("the operator acknowledges");
    await engineerApi.remove("/api/v1/simulation/faults");
    await frames.until(
      "the pump is repaired; the level recovers",
      async () => (await engineerApi.get<PlcStatus>("/api/v1/plc/status")).reset_permitted,
      300_000,
      SLOW_FRAME_EVERY_MS,
    );
    await engineerApi.post("/api/v1/simulation/speed", { speed_factor: RESTART_SPEED });
    await engineerApi.post("/api/v1/commands/reset", { operator_id: "readme" });
    await expect.poll(() => plcMode(operator)).toBe("AUTO");
    await frames.shot("the engineer resets the E-Stop: back to AUTO", true);
    await frames.until(
      "the unit ramps back to its load",
      async () => (await outputMw(operator)) > 150,
      600_000,
      SLOW_FRAME_EVERY_MS,
    );
    await frames.shot("back on load", true);
    frames.write();

    // Who did what, to the second — the audit log the admin reads after the demo.
    const admin = await (await browser.newContext()).newPage();
    await signIn(admin, "admin");
    await nav(admin, "Audit").click();
    const filters = admin.getByRole("form", { name: "Audit filters" });
    await filters.getByLabel("Path starts with").fill("/api/v1/commands");
    await filters.getByRole("button", { name: "Apply" }).click();
    await expect(
      admin.getByRole("region", { name: "Audit log" }).getByRole("row").nth(1),
    ).toBeVisible();
    await picture(admin, "audit");
    await admin.context().close();
  } finally {
    await nominal(engineerApi, 1);
    await operatorApi.close();
    await engineerApi.close();
    await operator.context().close();
    await engineer.context().close();
  }
});
