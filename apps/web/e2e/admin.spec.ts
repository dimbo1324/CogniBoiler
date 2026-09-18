import { expect, test } from "@playwright/test";
import { randomBytes } from "node:crypto";

import { signIn } from "./support";

// One account the checks own. It is created on the first run and left blocked, with a
// password nobody knows, after every run.
const E2E_USER = "e2e-user";

function freshPassword(): string {
  return randomBytes(18).toString("base64url");
}

test("an admin creates, promotes, blocks and unblocks a user; blocking closes sign-in", async ({
  page,
  browser,
}) => {
  await signIn(page, "admin");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Users" })
    .click();
  const users = page.getByRole("region", { name: "Users" });
  const row = page.getByTestId(`user-${E2E_USER}`);
  await expect(page.getByTestId("user-admin")).toBeVisible();

  if ((await row.count()) === 0) {
    const form = page.getByRole("form", { name: "Create a user" });
    await form.getByLabel("Username").fill(E2E_USER);
    await form.getByLabel("Initial password").fill(freshPassword());
    await form.getByLabel("Role").selectOption("viewer");
    await form.getByRole("button", { name: "Create" }).click();
    await expect(users).toContainText(`Created ${E2E_USER} as viewer.`);
  }

  const password = freshPassword();
  await row.getByLabel(`New password for ${E2E_USER}`).fill(password);
  await row.getByRole("button", { name: "Reset password…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Reset password" }).click();
  await expect(users).toContainText(`The password of ${E2E_USER} was reset.`);

  if ((await row.getByRole("button", { name: "Unblock…" }).count()) > 0) {
    await row.getByRole("button", { name: "Unblock…" }).click();
    await page.getByRole("dialog").getByRole("button", { name: "Unblock" }).click();
    await expect(row).toContainText("active");
  }

  await row.getByLabel(`Role of ${E2E_USER}`).selectOption("operator");
  await row.getByRole("button", { name: "Apply…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Make operator" }).click();
  await expect(users).toContainText(`${E2E_USER} is now operator.`);

  const other = await browser.newContext();
  const userPage = await other.newPage();
  await userPage.goto("/");
  await userPage.getByLabel("Username").fill(E2E_USER);
  await userPage.getByLabel("Password").fill(password);
  await userPage.getByRole("button", { name: "Sign in" }).click();
  await expect(userPage.getByLabel("Signed in as")).toContainText(`${E2E_USER} (operator)`);

  await row.getByRole("button", { name: "Block…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Block" }).click();
  await expect(row).toContainText("blocked");

  await userPage
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Alarms" })
    .click();
  await expect(userPage.getByRole("button", { name: "Sign in" })).toBeVisible({ timeout: 45_000 });
  await userPage.getByLabel("Username").fill(E2E_USER);
  await userPage.getByLabel("Password").fill(password);
  await userPage.getByRole("button", { name: "Sign in" }).click();
  await expect(userPage.getByRole("alert")).toHaveText("Invalid username or password.");
  await other.close();

  await row.getByLabel(`Role of ${E2E_USER}`).selectOption("viewer");
  await row.getByRole("button", { name: "Apply…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Make viewer" }).click();
  await expect(users).toContainText(`${E2E_USER} is now viewer.`);
  await page.screenshot({ path: "e2e-results/users.png", fullPage: true });
});

test("an admin cannot block their own account", async ({ page }) => {
  await signIn(page, "admin");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Users" })
    .click();
  await expect(
    page.getByTestId("user-admin").getByRole("button", { name: "Block…" }),
  ).toBeDisabled();
});

test("the audit log filters by user and path and shows refusals", async ({ page }) => {
  await signIn(page, "admin");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Audit" })
    .click();
  const filters = page.getByRole("form", { name: "Audit filters" });
  await filters.getByLabel("User").fill("admin");
  await filters.getByLabel("Path starts with").fill("/api/v1/users");
  await filters.getByRole("button", { name: "Apply" }).click();
  const log = page.getByRole("region", { name: "Audit log" });
  await expect(log.getByRole("row").nth(1)).toContainText("/api/v1/users");
  await expect(log.getByRole("row").nth(1)).toContainText("admin");
  await page.screenshot({ path: "e2e-results/audit.png", fullPage: true });

  await filters.getByLabel("User").fill("");
  await filters.getByLabel("Path starts with").fill("/auth/login");
  await filters.getByLabel("Refusals only").check();
  await filters.getByRole("button", { name: "Apply" }).click();
  await expect(log.getByRole("row").nth(1)).toContainText("refused: invalid credentials");
});

test("the platform screen shows every service and the telemetry age", async ({ page }) => {
  await signIn(page, "viewer");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Platform" })
    .click();
  await expect(page.getByTestId("platform-status")).toContainText("Ready");
  for (const name of [
    "database",
    "physics-engine",
    "plc-controller",
    "alert-manager",
    "historian",
  ]) {
    await expect(page.getByTestId(`component-${name}`)).toContainText("up");
  }
  await expect(page.getByRole("region", { name: "Live data" })).toContainText("s ago");
});
