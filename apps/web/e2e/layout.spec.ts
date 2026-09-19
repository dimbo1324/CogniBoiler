// The console on the screens a demo meets: a tablet held either way and a projector. Every
// screen an admin can open must fit the width without a horizontal page scroll, with its
// navigation and its first heading on screen. Screenshots land in e2e-results for review.

import { expect, test, type Page } from "@playwright/test";

import { signIn } from "./support";

const VIEWPORTS = [
  { name: "tablet-portrait", width: 768, height: 1024 },
  { name: "tablet-landscape", width: 1024, height: 768 },
  { name: "projector", width: 1280, height: 720 },
] as const;

const SCREENS = [
  "Overview",
  "Trends",
  "Alarms",
  "Control",
  "Engineer",
  "Audit",
  "Users",
  "Platform",
] as const;

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

for (const viewport of VIEWPORTS) {
  test(`every screen fits a ${viewport.name} (${String(viewport.width)}×${String(viewport.height)})`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto("/");
    expect(await horizontalOverflow(page), "sign-in screen").toBeLessThanOrEqual(0);
    await page.screenshot({ path: `e2e-results/layout-${viewport.name}-sign-in.png` });

    await signIn(page, "admin");
    const navigation = page.getByRole("navigation", { name: "Screens" });
    for (const name of SCREENS) {
      const link = navigation.getByRole("link", { name, exact: true });
      await expect(link, `${name} link`).toBeInViewport();
      await link.click();
      await expect(page.getByRole("heading").first()).toBeVisible();
      await page.waitForLoadState("networkidle");
      expect(await horizontalOverflow(page), `${name} screen`).toBeLessThanOrEqual(0);
      await page.screenshot({
        path: `e2e-results/layout-${viewport.name}-${name.toLowerCase()}.png`,
        fullPage: true,
      });
    }
  });
}
