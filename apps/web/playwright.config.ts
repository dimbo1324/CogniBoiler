import { defineConfig, devices } from "@playwright/test";

// End-to-end checks of the console against a running stack. CONSOLE_URL points at the
// console: the Vite dev server by default (started here if it is not running), or the
// nginx entry of the Compose stack. Demo passwords come from the repository's .env.
const consoleUrl = process.env.CONSOLE_URL ?? "http://127.0.0.1:5173";
const useDevServer = !process.env.CONSOLE_URL;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e-results",
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never", outputFolder: "e2e-report" }]]
    : "list",
  use: {
    baseURL: consoleUrl,
    ignoreHTTPSErrors: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1400, height: 900 } },
    },
  ],
  webServer: useDevServer
    ? {
        command: "pnpm run dev",
        url: consoleUrl,
        reuseExistingServer: true,
        timeout: 60_000,
      }
    : undefined,
});
