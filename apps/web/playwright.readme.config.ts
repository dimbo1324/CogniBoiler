import { defineConfig, devices } from "@playwright/test";

// The README's pictures, taken from a running stack by `readme-media` (see readme/). Kept
// apart from playwright.config.ts on purpose: this records, it does not check, and neither
// `console-e2e` nor CI runs it. README_MEDIA_DIR is where the raw captures go.
// IPv4 loopback, as every published port of the stack binds it.
const consoleUrl = process.env.CONSOLE_URL ?? "http://127.0.0.1:8080";

export default defineConfig({
  testDir: "./readme",
  outputDir: "./readme-results/playwright",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  // The demo plays at ten times real speed, the restart at three: most of the run is the
  // unit ramping back to its load.
  timeout: 1_500_000,
  expect: { timeout: 30_000 },
  reporter: "list",
  use: {
    baseURL: consoleUrl,
    ignoreHTTPSErrors: true,
    screenshot: "only-on-failure",
    // No trace: it would record the demo passwords the capture signs in with.
    trace: "off",
  },
  projects: [
    {
      name: "readme",
      // One CSS pixel per image pixel keeps the files small; the size is a laptop's.
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 1,
        colorScheme: "dark",
      },
    },
  ],
});
