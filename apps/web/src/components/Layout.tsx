import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router";

import type { ConnectionState } from "../api/realtime";
import { useLive } from "../live/LiveProvider";
import { useRole, useSession, useUser } from "../session/SessionProvider";
import { can, type Permission } from "../session/roles";
import { applyTheme, nextTheme, storedTheme, watchSystemTheme, type ThemeChoice } from "../theme";
import { AlarmBanner } from "./AlarmBanner";
import { PlcModeBadge } from "./PlcModeBadge";
import { Badge, type BadgeTone } from "./ui/Badge";
import { Icon, type IconGlyph } from "./ui/Icon";
import {
  AlarmsIcon,
  AuditIcon,
  BrandIcon,
  ControlIcon,
  EngineerIcon,
  LiveIcon,
  OfflineIcon,
  OverviewIcon,
  PendingIcon,
  PlatformIcon,
  SignOutIcon,
  ThemeDarkIcon,
  ThemeLightIcon,
  ThemeSystemIcon,
  TrendsIcon,
  UsersIcon,
} from "./ui/icons";

interface NavItem {
  to: string;
  label: string;
  glyph: IconGlyph;
  permission?: Permission;
}

export const NAV_ITEMS: readonly NavItem[] = [
  { to: "/", label: "Overview", glyph: OverviewIcon },
  { to: "/trends", label: "Trends", glyph: TrendsIcon },
  { to: "/alarms", label: "Alarms", glyph: AlarmsIcon },
  { to: "/control", label: "Control", glyph: ControlIcon, permission: "set_load" },
  {
    to: "/engineer",
    label: "Engineer",
    glyph: EngineerIcon,
    permission: "control_simulation",
  },
  { to: "/audit", label: "Audit", glyph: AuditIcon, permission: "read_audit" },
  { to: "/users", label: "Users", glyph: UsersIcon, permission: "manage_users" },
  { to: "/platform", label: "Platform", glyph: PlatformIcon },
];

interface ConnectionLook {
  label: string;
  tone: BadgeTone;
  glyph: IconGlyph;
  spin?: boolean;
}

const CONNECTION: Record<ConnectionState, ConnectionLook> = {
  connecting: { label: "Connecting…", tone: "warn", glyph: PendingIcon, spin: true },
  live: { label: "Live", tone: "ok", glyph: LiveIcon },
  reconnecting: { label: "Reconnecting…", tone: "warn", glyph: PendingIcon, spin: true },
  stopped: { label: "Offline", tone: "warn", glyph: OfflineIcon },
};

const THEME: Record<ThemeChoice, { label: string; glyph: IconGlyph }> = {
  system: { label: "Theme: system", glyph: ThemeSystemIcon },
  dark: { label: "Theme: dark", glyph: ThemeDarkIcon },
  light: { label: "Theme: light", glyph: ThemeLightIcon },
};

export function Layout() {
  const { manager } = useSession();
  const user = useUser();
  const role = useRole();
  const live = useLive();
  const [theme, setTheme] = useState<ThemeChoice>(storedTheme);

  // While the operator follows the system, a change of the system's mode reaches the page.
  useEffect(() => {
    if (theme !== "system") {
      return;
    }
    return watchSystemTheme(() => {
      applyTheme("system");
    });
  }, [theme]);

  const connection = CONNECTION[live.connection];
  const appearance = THEME[theme];

  return (
    <>
      <header className="shell-header">
        <span className="brand">
          <Icon glyph={BrandIcon} />
          CogniBoiler
        </span>
        <nav className="shell-nav" aria-label="Screens">
          {NAV_ITEMS.filter((item) => !item.permission || can(role, item.permission)).map(
            (item) => (
              <NavLink key={item.to} to={item.to} end={item.to === "/"}>
                <Icon glyph={item.glyph} />
                {item.label}
              </NavLink>
            ),
          )}
        </nav>
        <div className="shell-status">
          <Badge
            tone={connection.tone}
            glyph={connection.glyph}
            spin={connection.spin}
            label="Live data connection"
          >
            {connection.label}
          </Badge>
          <PlcModeBadge plc={live.plc} />
          <span className="shell-user" aria-label="Signed in as">
            {user.username} <span className="muted">({user.role})</span>
          </span>
          <button
            type="button"
            className="quiet"
            onClick={() => {
              const next = nextTheme(theme);
              applyTheme(next);
              setTheme(next);
            }}
          >
            <Icon glyph={appearance.glyph} tone="muted" />
            {appearance.label}
          </button>
          <button
            type="button"
            className="quiet"
            onClick={() => {
              void manager.signOut();
            }}
          >
            <Icon glyph={SignOutIcon} tone="muted" />
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
