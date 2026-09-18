import { expect, test, type Page } from "@playwright/test";

import { signIn } from "./support";

async function screens(page: Page): Promise<string[]> {
  return page.getByRole("navigation", { name: "Screens" }).getByRole("link").allTextContents();
}

test("a viewer watches but sees no control, and a control screen says so", async ({ page }) => {
  await signIn(page, "viewer");
  expect(await screens(page)).toEqual(["Overview", "Trends", "Alarms", "Platform"]);
  await page.goto("/control");
  await expect(page.getByRole("status")).toHaveText(
    "This screen is not available to the viewer role.",
  );
  await page.goto("/users");
  await expect(page.getByRole("status")).toHaveText(
    "This screen is not available to the viewer role.",
  );
});

test("an operator controls load, mode and valves but not setpoints or the trip reset", async ({
  page,
}) => {
  await signIn(page, "operator");
  expect(await screens(page)).toEqual(["Overview", "Trends", "Alarms", "Control", "Platform"]);
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Control" })
    .click();
  await expect(page.getByRole("region", { name: "Load" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Mode" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Manual valves" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Setpoints" })).toHaveCount(0);
  await page.screenshot({ path: "e2e-results/control-operator.png", fullPage: true });
});

test("an engineer also has setpoints and the engineer panel", async ({ page }) => {
  await signIn(page, "engineer");
  expect(await screens(page)).toEqual([
    "Overview",
    "Trends",
    "Alarms",
    "Control",
    "Engineer",
    "Platform",
  ]);
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Control" })
    .click();
  await expect(page.getByRole("region", { name: "Setpoints" })).toBeVisible();
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Engineer" })
    .click();
  await expect(page.getByRole("region", { name: "Faults" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Scenarios" })).toContainText(
    "Feedwater pump failure drill",
  );
  await page.screenshot({ path: "e2e-results/engineer.png", fullPage: true });
});

test("an admin has the audit log and user administration", async ({ page }) => {
  await signIn(page, "admin");
  expect(await screens(page)).toEqual([
    "Overview",
    "Trends",
    "Alarms",
    "Control",
    "Engineer",
    "Audit",
    "Users",
    "Platform",
  ]);
});
