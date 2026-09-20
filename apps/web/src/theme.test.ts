import { afterEach, describe, expect, it, vi } from "vitest";

import { Horn } from "./alarms/horn";
import {
  DEFAULT_THEME,
  applyTheme,
  nextTheme,
  resolveTheme,
  storedTheme,
  systemTheme,
  watchSystemTheme,
} from "./theme";
import { plantState } from "./test/fixtures";
import { DEFAULT_TREND_IDS, TREND_PARAMETERS, trendParameter } from "./trends/parameters";

afterEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
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

class FakeAudioContext {
  static created = 0;
  static oscillators = 0;
  currentTime = 0;
  destination = {};
  constructor() {
    FakeAudioContext.created += 1;
  }
  createGain() {
    return { gain: { value: 0 }, connect: vi.fn() };
  }
  createOscillator() {
    FakeAudioContext.oscillators += 1;
    return { frequency: { value: 0 }, connect: vi.fn(), start: vi.fn(), stop: vi.fn() };
  }
}

describe("horn", () => {
  it("beeps now and every two seconds until stopped, with one audio context", () => {
    vi.useFakeTimers();
    FakeAudioContext.created = 0;
    FakeAudioContext.oscillators = 0;
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const horn = new Horn();
    horn.start();
    horn.start();
    expect(horn.sounding).toBe(true);
    expect(FakeAudioContext.oscillators).toBe(2);
    vi.advanceTimersByTime(4000);
    expect(FakeAudioContext.oscillators).toBe(6);
    horn.stop();
    expect(horn.sounding).toBe(false);
    vi.advanceTimersByTime(4000);
    expect(FakeAudioContext.oscillators).toBe(6);
    expect(FakeAudioContext.created).toBe(1);
  });

  it("stays silent without Web Audio or when the browser refuses", () => {
    vi.useFakeTimers();
    vi.stubGlobal("AudioContext", undefined);
    const quiet = new Horn();
    quiet.start();
    quiet.stop();
    vi.stubGlobal(
      "AudioContext",
      class {
        constructor() {
          throw new Error("no audio device");
        }
      },
    );
    const refused = new Horn();
    expect(() => {
      refused.start();
    }).not.toThrow();
    refused.stop();
  });
});

describe("trend parameters", () => {
  it("have unique ids and the defaults exist", () => {
    const ids = TREND_PARAMETERS.map((parameter) => parameter.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of DEFAULT_TREND_IDS) {
      expect(trendParameter(id)).toBeDefined();
    }
    expect(trendParameter("nowhere")).toBeUndefined();
  });

  it("read a live snapshot and convert to the unit shown", () => {
    const state = plantState();
    for (const parameter of TREND_PARAMETERS) {
      const shown = parameter.display(parameter.live(state));
      expect(Number.isFinite(shown), parameter.id).toBe(true);
      expect(parameter.history.field.length).toBeGreaterThan(0);
    }
    const power = trendParameter("electrical_power");
    expect(power?.display(power.live(state))).toBeCloseTo(300.0, 1);
    expect(power?.unit).toBe("MW");
  });
});
