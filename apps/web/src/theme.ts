// The operator's theme choice: the system's, or light or dark. Kept in localStorage when the
// browser allows it; without storage the choice lasts until the page reloads.

export type ThemeChoice = "system" | "light" | "dark";

const STORAGE_KEY = "cogniboiler.theme";

export function storedTheme(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", choice);
  }
  try {
    if (choice === "system") {
      window.localStorage.removeItem(STORAGE_KEY);
    } else {
      window.localStorage.setItem(STORAGE_KEY, choice);
    }
  } catch {
    // Storage is unavailable (private window, blocked site data): the choice still applies.
  }
}

export function nextTheme(choice: ThemeChoice): ThemeChoice {
  return choice === "system" ? "dark" : choice === "dark" ? "light" : "system";
}
