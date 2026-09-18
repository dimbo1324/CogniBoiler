// Helpers shared by the end-to-end specs. Passwords are the demo users' from the repository's
// .env (written by dev-secrets), or from DEMO_*_PASSWORD in the environment, as in CI.

import { expect, type Page } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type DemoRole = "viewer" | "operator" | "engineer" | "admin";

const ENV_FILE = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", ".env");

function envFileValues(): Record<string, string> {
  if (!existsSync(ENV_FILE)) {
    return {};
  }
  const values: Record<string, string> = {};
  for (const line of readFileSync(ENV_FILE, "utf-8").split(/\r?\n/u)) {
    const match = /^(DEMO_[A-Z]+_PASSWORD)=(.*)$/u.exec(line);
    if (match?.[1] !== undefined && match[2] !== undefined) {
      values[match[1]] = match[2].trim().replace(/^"(.*)"$/u, "$1");
    }
  }
  return values;
}

export function demoPassword(role: DemoRole): string {
  const key = `DEMO_${role.toUpperCase()}_PASSWORD`;
  const password = process.env[key] ?? envFileValues()[key];
  if (!password) {
    throw new Error(`${key} is not set: run dev-secrets, or export it`);
  }
  return password;
}

export async function signIn(page: Page, role: DemoRole): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Username").fill(role);
  await page.getByLabel("Password").fill(demoPassword(role));
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByLabel("Signed in as")).toContainText(role);
}

export async function signOut(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
}
