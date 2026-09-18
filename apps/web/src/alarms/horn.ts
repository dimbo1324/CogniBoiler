// The audible alarm: a short two-tone beep every two seconds while an unacknowledged critical
// alarm stands and the operator has not silenced it. Generated with Web Audio, so there is
// no sound file to ship. Browsers only allow sound after a user gesture; signing in is one.

const PERIOD_MS = 2000;

export class Horn {
  private context: AudioContext | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  get sounding(): boolean {
    return this.timer !== null;
  }

  start(): void {
    if (this.timer !== null) {
      return;
    }
    this.beep();
    this.timer = setInterval(() => {
      this.beep();
    }, PERIOD_MS);
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private beep(): void {
    const AudioContextClass = typeof window === "undefined" ? undefined : window.AudioContext;
    if (AudioContextClass === undefined) {
      return;
    }
    try {
      this.context ??= new AudioContextClass();
      const context = this.context;
      const gain = context.createGain();
      gain.gain.value = 0.08;
      gain.connect(context.destination);
      [880, 660].forEach((frequency, index) => {
        const oscillator = context.createOscillator();
        oscillator.frequency.value = frequency;
        oscillator.connect(gain);
        const start = context.currentTime + index * 0.18;
        oscillator.start(start);
        oscillator.stop(start + 0.15);
      });
    } catch {
      // No audio device or the browser refused: the flashing banner still alerts.
    }
  }
}
