// The five-minute demo of VISION §7, played through the console by three people at once:
// an operator, an engineer and an admin, each in their own browser session. The simulation
// runs ten times faster so the trip and the recovery take a minute, not ten.

import { expect, request, test, type Browser, type Page } from "@playwright/test";

import { demoPassword, signIn, type DemoRole } from "./support";

const DEMO_SPEED = 10;

async function personPage(browser: Browser, role: DemoRole): Promise<Page> {
  const context = await browser.newContext();
  const page = await context.newPage();
  await signIn(page, role);
  return page;
}

function screen(page: Page, name: string) {
  return page.getByRole("navigation", { name: "Screens" }).getByRole("link", { name, exact: true });
}

async function confirm(page: Page, button: string | RegExp): Promise<void> {
  await page.getByRole("dialog").getByRole("button", { name: button }).click();
}

async function outputMw(page: Page): Promise<number> {
  const text = (await page.getByTestId("mimic-power").textContent()) ?? "";
  return Number(/(-?\d+(?:\.\d+)?)/u.exec(text)?.[1] ?? Number.NaN);
}

/** Put the simulation back to real time whatever happened, through the API. */
async function restoreRealTime(baseURL: string | undefined): Promise<void> {
  const api = await request.newContext({ baseURL, ignoreHTTPSErrors: true });
  try {
    const login = await api.post("/auth/login", {
      data: { username: "engineer", password: demoPassword("engineer") },
    });
    const { access_token: token } = (await login.json()) as { access_token: string };
    await api.post("/api/v1/simulation/speed", {
      data: { speed_factor: 1 },
      headers: { Authorization: `Bearer ${token}` },
    });
  } finally {
    await api.dispose();
  }
}

test("the demo scenario: load change, pump failure, trip, acknowledgement, reset, audit", async ({
  browser,
  baseURL,
}) => {
  test.setTimeout(420_000);
  const engineer = await personPage(browser, "engineer");
  const operator = await personPage(browser, "operator");
  try {
    // Setup: the unit at 250 MW, simulated ten times faster.
    await screen(engineer, "Engineer").click();
    const scenarios = engineer.getByRole("region", { name: "Scenarios" });
    await scenarios
      .getByRole("row")
      .filter({ has: engineer.getByText("Nominal load", { exact: true }) })
      .getByRole("button", { name: "Load…" })
      .click();
    await confirm(engineer, "Load scenario");
    await expect(engineer.getByRole("region", { name: "Last action" })).toContainText("Accepted.");
    await engineer.getByLabel("Speed").selectOption(String(DEMO_SPEED));
    await engineer.getByRole("button", { name: "Set speed" }).click();
    await expect(engineer.getByTestId("simulation-state")).toContainText(
      `running at ${String(DEMO_SPEED)}×`,
    );

    // 0:30 — the operator sees the unit at 250 MW in AUTO.
    await expect.poll(() => outputMw(operator), { timeout: 30_000 }).toBeGreaterThan(240);
    expect(await outputMw(operator)).toBeLessThan(260);
    await expect(operator.getByRole("banner").getByLabel("PLC mode")).toHaveText("AUTO");
    await screen(operator, "Alarms").click();
    const acknowledgeAll = operator.getByRole("button", { name: /Acknowledge all/u });
    if (await acknowledgeAll.isEnabled()) {
      await acknowledgeAll.click();
    }

    // 1:00 — the operator raises the load to 300 MW and the unit follows.
    await screen(operator, "Control").click();
    await operator.getByLabel("New demand [MW]").fill("300");
    await operator
      .getByRole("region", { name: "Load" })
      .getByRole("button", { name: "Set load…" })
      .click();
    await confirm(operator, "Set 300.0 MW");
    await expect(operator.getByRole("region", { name: "Last command" })).toContainText("Accepted.");
    await screen(operator, "Overview").click();
    await expect.poll(() => outputMw(operator), { timeout: 60_000 }).toBeGreaterThan(290);
    await operator.screenshot({ path: "e2e-results/demo-1-300mw.png", fullPage: true });

    // 2:00 — the engineer fails the feedwater pump.
    const faults = engineer.getByRole("region", { name: "Faults" });
    await faults.getByRole("combobox").first().selectOption("feedwater_pump_failure");
    await faults.getByRole("button", { name: "Inject…" }).click();
    await confirm(engineer, "Inject fault");
    await expect(engineer.getByTestId("fault-feedwater_pump_failure")).toBeVisible();

    // 2:20 — the level falls, the interlock trips the unit, a critical alarm flashes.
    await expect(operator.getByRole("banner").getByLabel("PLC mode")).toHaveText("E-STOP", {
      timeout: 120_000,
    });
    const banner = operator.getByRole("alert").filter({ hasText: "critical" });
    await expect(banner).toHaveClass(/flashing/u);
    await operator.screenshot({ path: "e2e-results/demo-2-trip.png", fullPage: true });

    // 3:00 — acknowledge, clear the fault, reset once the cause has cleared.
    await screen(operator, "Alarms").click();
    await operator.getByRole("button", { name: /Acknowledge all/u }).click();
    await expect(operator.getByRole("alert").filter({ hasText: "critical" })).toHaveCount(0);

    await engineer
      .getByTestId("fault-feedwater_pump_failure")
      .getByRole("button", { name: "Clear…" })
      .click();
    await confirm(engineer, "Clear fault");
    await expect(engineer.getByTestId("fault-feedwater_pump_failure")).toHaveCount(0);

    await screen(engineer, "Control").click();
    const reset = engineer.getByRole("region", { name: "Emergency stop reset" });
    await expect(reset).toContainText("The cause has cleared", { timeout: 180_000 });
    await reset.getByRole("button", { name: "Reset E-Stop…" }).click();
    await confirm(engineer, "Reset E-Stop");
    await expect(engineer.getByRole("region", { name: "Last command" })).toContainText("Accepted.");
    await expect(operator.getByRole("banner").getByLabel("PLC mode")).toHaveText("AUTO");

    // The unit goes back on load in AUTO.
    await screen(operator, "Overview").click();
    await expect.poll(() => outputMw(operator), { timeout: 120_000 }).toBeGreaterThan(50);
    await operator.screenshot({ path: "e2e-results/demo-3-recovery.png", fullPage: true });

    // 4:10 — the admin reads who changed the load and who reset the trip, and when.
    const admin = await personPage(browser, "admin");
    await screen(admin, "Audit").click();
    const filters = admin.getByRole("form", { name: "Audit filters" });
    await filters.getByLabel("Path starts with").fill("/api/v1/commands");
    await filters.getByRole("button", { name: "Apply" }).click();
    const log = admin.getByRole("region", { name: "Audit log" });
    await expect(
      log.getByRole("row").filter({ hasText: "/api/v1/commands/reset" }).first(),
    ).toContainText("engineer");
    await expect(
      log.getByRole("row").filter({ hasText: "/api/v1/commands/load" }).first(),
    ).toContainText("operator");
    await admin.screenshot({ path: "e2e-results/demo-4-audit.png", fullPage: true });
    await admin.context().close();
  } finally {
    await restoreRealTime(baseURL);
    await operator.context().close();
    await engineer.context().close();
  }
});
