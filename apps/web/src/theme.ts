// The console is dark by default (owner decision 2026-09-20): a control room is dim, and a
// white screen behind a mimic is glare. The operator may still pick light, or follow the
// system. The choice is kept in localStorage when the browser allows it; without storage it
// lasts until the page reloads.
//
// Only the RESOLVED theme reaches the page, as data-theme on <html>. The stylesheet then
// needs one palette per theme instead of one per theme and per way of choosing it.

export type ThemeChoice = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "cogniboiler.theme";
const LIGHT_QUERY = "(prefers-color-scheme: light)";

export const DEFAULT_THEME: ThemeChoice = "dark";
/** The order of the button: dark → light → system → dark. */
const CYCLE: readonly ThemeChoice[] = ["dark", "light", "system"];

function isChoice(value: string | null): value is ThemeChoice {
  return value === "system" || value === "light" || value === "dark";
}

export function storedTheme(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return isChoice(value) ? value : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

/** What the system asks for. Anything but an explicit request for light means dark. */
export function systemTheme(): ResolvedTheme {
  try {
    return window.matchMedia(LIGHT_QUERY).matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  return choice === "system" ? systemTheme() : choice;
}

export function applyTheme(choice: ThemeChoice): ResolvedTheme {
  const resolved = resolveTheme(choice);
  const root = document.documentElement;
  root.setAttribute("data-theme", resolved);
  try {
    window.localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // Storage is unavailable (private window, blocked site data): the choice still applies.
  }
  return resolved;
}

export function nextTheme(choice: ThemeChoice): ThemeChoice {
  const index = CYCLE.indexOf(choice);
  return CYCLE[(index + 1) % CYCLE.length] ?? DEFAULT_THEME;
}

/**
 * Follow the system while the operator's choice is "system". Returns the unsubscribe.
 * A browser without matchMedia (or an old one with only addListener) simply never changes.
 */
export function watchSystemTheme(onChange: (resolved: ResolvedTheme) => void): () => void {
  let media: MediaQueryList;
  try {
    media = window.matchMedia(LIGHT_QUERY);
  } catch {
    return () => undefined;
  }
  const listener = () => {
    onChange(systemTheme());
  };
  if (typeof media.addEventListener !== "function") {
    return () => undefined;
  }
  media.addEventListener("change", listener);
  return () => {
    media.removeEventListener("change", listener);
  };
}
