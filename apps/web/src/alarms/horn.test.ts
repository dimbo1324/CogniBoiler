import { afterEach, describe, expect, it, vi } from "vitest";

import { Horn } from "./horn";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

class FakeAudioContext {
  static created = 0;
  static oscillators = 0;
  static closed = 0;
  currentTime = 0;
  destination = {};
  constructor() {
    FakeAudioContext.created += 1;
  }
  createGain() {
    return { gain: { value: 0 }, connect: vi.fn() };
  }
  close() {
    FakeAudioContext.closed += 1;
    return Promise.resolve();
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

  it("closes its audio context when disposed, and stops beeping", () => {
    vi.useFakeTimers();
    FakeAudioContext.created = 0;
    FakeAudioContext.oscillators = 0;
    FakeAudioContext.closed = 0;
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const horn = new Horn();
    horn.start();
    horn.dispose();
    expect(FakeAudioContext.closed).toBe(1);
    expect(horn.sounding).toBe(false);
    vi.advanceTimersByTime(4000);
    expect(FakeAudioContext.oscillators).toBe(2);
    horn.dispose();
    expect(FakeAudioContext.closed).toBe(1);
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
