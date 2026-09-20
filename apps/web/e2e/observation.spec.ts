import { expect, test } from "@playwright/test";

import { signIn, signOut } from "./support";

test("a wrong password is refused without saying whether the account exists", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Username").fill("nobody-e2e");
  await page.getByLabel("Password").fill("not-the-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("alert")).toHaveText("Invalid username or password.");
});

test("the operator sees the live mimic, and a reload keeps the session", async ({ page }) => {
  const policyViolations: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error" && message.text().includes("Content Security Policy")) {
      policyViolations.push(message.text());
    }
  });
  await signIn(page, "operator");
  await expect(page.getByLabel("Live data connection")).toHaveText("Live");
  await expect(page.getByTestId("mimic-power")).toContainText("MW");
  await expect(page.getByTestId("mimic-drum-pressure")).toContainText("bar");
  await expect(page.getByRole("banner").getByLabel("PLC mode")).toBeVisible();
  await page.screenshot({ path: "e2e-results/overview.png", fullPage: true });
  expect(policyViolations).toEqual([]);

  await page.reload();
  await expect(page.getByLabel("Signed in as")).toContainText("operator");
  await signOut(page);
  await page.reload();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("trends draw live values and recorded history with KPIs", async ({ page }) => {
  await signIn(page, "viewer");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Trends" })
    .click();
  await expect(page.getByTestId("trend-electrical_power")).toBeVisible();
  await page.getByRole("button", { name: "1 h" }).click();
  await expect(page.getByTestId("kpis")).toContainText("kJ/kWh");
  await expect(page.getByTestId("trend-electrical_power").locator("canvas")).toBeVisible();
  await page.screenshot({ path: "e2e-results/trends.png", fullPage: true });
});

test("the alarm screen lists active alarms and history", async ({ page }) => {
  await signIn(page, "viewer");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Alarms" })
    .click();
  await expect(page.getByRole("region", { name: "Active alarms" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Acknowledge all/u })).toHaveCount(0);
  const loaded = page.waitForResponse((response) =>
    response.url().includes("/api/v1/alarms/history"),
  );
  await page.getByRole("tab", { name: "History" }).click();
  expect((await loaded).status()).toBe(200);
  await expect(page.getByRole("region", { name: "Alarm history" })).toBeVisible();
  await page.screenshot({ path: "e2e-results/alarms.png", fullPage: true });
});

test("the console is dark to begin with, and the choice survives a reload", async ({ page }) => {
  await signIn(page, "viewer");
  const html = page.locator("html");
  // Nobody chose anything yet: a control room console opens dark.
  await expect(html).toHaveAttribute("data-theme", "dark");
  await expect(page.getByTestId("mimic-power")).toContainText("MW");
  await page.screenshot({ path: "e2e-results/overview-dark.png", fullPage: true });

  await page.getByRole("button", { name: "Theme: dark" }).click();
  await expect(html).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", "light");
  await page.screenshot({ path: "e2e-results/overview-light.png", fullPage: true });

  // Following the system resolves to one of the two themes, never to no theme at all.
  await page.getByRole("button", { name: "Theme: light" }).click();
  await expect(html).toHaveAttribute("data-theme", /^(dark|light)$/u);
  await page.getByRole("button", { name: "Theme: system" }).click();
  await expect(html).toHaveAttribute("data-theme", "dark");
});
