import { useState } from "react";
import { NavLink, Outlet } from "react-router";

import type { ConnectionState } from "../api/realtime";
import { useLive } from "../live/LiveProvider";
import { useRole, useSession, useUser } from "../session/SessionProvider";
import { can, type Permission } from "../session/roles";
import { applyTheme, nextTheme, storedTheme, type ThemeChoice } from "../theme";
import { AlarmBanner } from "./AlarmBanner";
import { PlcModeBadge } from "./PlcModeBadge";

interface NavItem {
  to: string;
  label: string;
  permission?: Permission;
}

export const NAV_ITEMS: readonly NavItem[] = [
  { to: "/", label: "Overview" },
  { to: "/trends", label: "Trends" },
  { to: "/alarms", label: "Alarms" },
];

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  live: "Live",
  reconnecting: "Reconnecting…",
  stopped: "Offline",
};

const THEME_LABEL: Record<ThemeChoice, string> = {
  system: "Theme: system",
  dark: "Theme: dark",
  light: "Theme: light",
};

export function Layout() {
  const { manager } = useSession();
  const user = useUser();
  const role = useRole();
  const live = useLive();
  const [theme, setTheme] = useState<ThemeChoice>(storedTheme);

  return (
    <>
      <header className="shell-header">
        <span className="brand">CogniBoiler</span>
        <nav className="shell-nav" aria-label="Screens">
          {NAV_ITEMS.filter((item) => !item.permission || can(role, item.permission)).map(
            (item) => (
              <NavLink key={item.to} to={item.to} end={item.to === "/"}>
                {item.label}
              </NavLink>
            ),
          )}
        </nav>
        <div className="shell-status">
          <span
            className={`badge ${live.connection === "live" ? "ok" : "warn"}`}
            aria-label="Live data connection"
          >
            {CONNECTION_LABEL[live.connection]}
          </span>
          <PlcModeBadge plc={live.plc} />
          <span aria-label="Signed in as">
            {user.username} <span className="muted">({user.role})</span>
          </span>
          <button
            type="button"
            onClick={() => {
              const next = nextTheme(theme);
              applyTheme(next);
              setTheme(next);
            }}
          >
            {THEME_LABEL[theme]}
          </button>
          <button
            type="button"
            onClick={() => {
              void manager.signOut();
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      <AlarmBanner />
      <main className="page">
        <Outlet />
      </main>
    </>
  );
}
