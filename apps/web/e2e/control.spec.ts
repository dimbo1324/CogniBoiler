import { expect, test } from "@playwright/test";

import { signIn } from "./support";

test("a load change asks first, reaches the PLC, and a cancel sends nothing", async ({ page }) => {
  await signIn(page, "operator");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Control" })
    .click();
  const load = page.getByRole("region", { name: "Load" });
  const demand = page.getByLabel("New demand [MW]");
  const original = await demand.inputValue();
  const target = String(Number(original) === 290 ? 280 : 290);

  await demand.fill(target);
  await load.getByRole("button", { name: "Set load…" }).click();
  const dialog = page.getByRole("dialog", { name: "Change the load demand" });
  await expect(dialog).toContainText(`${target}.0 MW`);
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Last command" })).toHaveCount(0);

  await load.getByRole("button", { name: "Set load…" }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: `Set ${target}.0 MW` })
    .click();
  await expect(page.getByRole("region", { name: "Last command" })).toContainText("Accepted.");
  await expect(load).toContainText(`Demand ${target}.0 MW`);

  await demand.fill(original);
  await load.getByRole("button", { name: "Set load…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: /^Set / }).click();
  await expect(load).toContainText(`Demand ${original}.0 MW`);
});

test("values outside the gateway's limits cannot be sent", async ({ page }) => {
  await signIn(page, "engineer");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("link", { name: "Control" })
    .click();
  await page.getByLabel("New demand [MW]").fill("350");
  await expect(
    page.getByRole("region", { name: "Load" }).getByRole("button", { name: "Set load…" }),
  ).toBeDisabled();
  await page.getByLabel("Drum pressure [bar]").fill("200");
  await expect(
    page.getByRole("region", { name: "Setpoints" }).getByRole("button", { name: "Apply…" }),
  ).toBeDisabled();
});
