import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_THEME,
  applyTheme,
  nextTheme,
  resolveTheme,
  storedTheme,
  systemTheme,
  watchSystemTheme,
} from "./theme";

afterEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("theme", () => {
  it("is dark until the operator chooses otherwise", () => {
    expect(DEFAULT_THEME).toBe("dark");
    expect(storedTheme()).toBe("dark");
    expect(applyTheme(storedTheme())).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("cycles dark, light, system", () => {
    expect(nextTheme("dark")).toBe("light");
    expect(nextTheme("light")).toBe("system");
    expect(nextTheme("system")).toBe("dark");
  });

  it("keeps the choice and puts the resolved theme on the page", () => {
    applyTheme("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(storedTheme()).toBe("light");
    applyTheme("system");
    expect(storedTheme()).toBe("system");
    // The stub of jsdom matches no media query, so the system asks for nothing and the
    // console stays dark — which is what an unknown preference should give.
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("follows the system only while that is the choice", () => {
    const listeners: (() => void)[] = [];
    let light = true;
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query === "(prefers-color-scheme: light)" && light,
      media: query,
      addEventListener: (_: string, listener: () => void) => listeners.push(listener),
      removeEventListener: (_: string, listener: () => void) => {
        listeners.splice(listeners.indexOf(listener), 1);
      },
    }));
    expect(systemTheme()).toBe("light");
    expect(resolveTheme("system")).toBe("light");
    expect(resolveTheme("dark")).toBe("dark");

    const seen: string[] = [];
    const stop = watchSystemTheme((resolved) => seen.push(resolved));
    light = false;
    listeners.forEach((listener) => {
      listener();
    });
    expect(seen).toEqual(["dark"]);
    stop();
    expect(listeners).toHaveLength(0);
  });

  it("ignores a stored value it does not know", () => {
    window.localStorage.setItem("cogniboiler.theme", "purple");
    expect(storedTheme()).toBe("dark");
  });

  it("still applies when storage is refused", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(applyTheme("light")).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(storedTheme()).toBe("dark");
  });

  it("survives a browser without matchMedia at all", () => {
    vi.stubGlobal("matchMedia", () => {
      throw new Error("not implemented");
    });
    expect(systemTheme()).toBe("dark");
    expect(watchSystemTheme(() => undefined)()).toBeUndefined();
    vi.stubGlobal("matchMedia", (query: string) => ({ matches: false, media: query }));
    expect(watchSystemTheme(() => undefined)()).toBeUndefined();
  });
});
