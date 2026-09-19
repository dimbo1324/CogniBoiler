import { afterEach, describe, expect, it, vi } from "vitest";

import { Horn } from "./alarms/horn";
import { applyTheme, nextTheme, storedTheme } from "./theme";
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
  it("cycles system, dark, light", () => {
    expect(nextTheme("system")).toBe("dark");
    expect(nextTheme("dark")).toBe("light");
    expect(nextTheme("light")).toBe("system");
  });

  it("is kept and applied, and the system choice forgets it", () => {
    applyTheme("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(storedTheme()).toBe("dark");
    applyTheme("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(storedTheme()).toBe("system");
  });

  it("ignores a stored value it does not know", () => {
    window.localStorage.setItem("cogniboiler.theme", "purple");
    expect(storedTheme()).toBe("system");
  });

  it("still applies when storage is refused", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    applyTheme("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(storedTheme()).toBe("system");
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
