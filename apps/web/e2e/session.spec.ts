// How a session ends in the console: closed by an administrator while in use, gone from the
// browser, or signed out. The console must return to sign-in and say why when there is a
// reason worth telling, without the user doing anything.

import { expect, test } from "@playwright/test";

import { signIn } from "./support";

test("a session an administrator closes returns the console to sign-in with the reason", async ({
  page,
  browser,
}) => {
  const operatorContext = await browser.newContext();
  const operator = await operatorContext.newPage();
  await signIn(operator, "operator");

  await signIn(page, "admin");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Users" })
    .click();
  const row = page.getByTestId("user-operator");
  await row.getByRole("button", { name: "Sign out everywhere…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Sign out everywhere" }).click();

  await expect(operator.getByRole("status")).toHaveText(
    "Your session was closed or your account changed. Sign in again.",
    { timeout: 45_000 },
  );
  await expect(operator.getByRole("button", { name: "Sign in" })).toBeVisible();
  await operator.reload();
  await expect(operator.getByRole("button", { name: "Sign in" })).toBeVisible();
  await operatorContext.close();
});

test("a session the browser no longer holds asks to sign in again after a reload", async ({
  page,
  context,
}) => {
  await signIn(page, "viewer");
  await context.clearCookies();
  await page.reload();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  await expect(page.getByLabel("Signed in as")).toHaveCount(0);
});

test("signing out ends the session, and a reload does not bring it back", async ({ page }) => {
  await signIn(page, "engineer");
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
  await expect(page.getByLabel("Signed in as")).toHaveCount(0);
});
