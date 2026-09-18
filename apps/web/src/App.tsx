import { useState } from "react";
import { createBrowserRouter, Navigate, RouterProvider, type RouteObject } from "react-router";

import { Layout } from "./components/Layout";
import { LiveProvider } from "./live/LiveProvider";
import { RequirePermission } from "./components/RequirePermission";
import { AlarmsScreen } from "./screens/AlarmsScreen";
import { AuditScreen } from "./screens/AuditScreen";
import { ControlScreen } from "./screens/ControlScreen";
import { EngineerScreen } from "./screens/EngineerScreen";
import { LoginScreen } from "./screens/LoginScreen";
import { OverviewScreen } from "./screens/OverviewScreen";
import { PlatformScreen } from "./screens/PlatformScreen";
import { TrendsScreen } from "./screens/TrendsScreen";
import { UsersScreen } from "./screens/UsersScreen";
import { SessionProvider, useSession } from "./session/SessionProvider";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <OverviewScreen /> },
      { path: "trends", element: <TrendsScreen /> },
      { path: "alarms", element: <AlarmsScreen /> },
      {
        path: "control",
        element: (
          <RequirePermission permission="set_load">
            <ControlScreen />
          </RequirePermission>
        ),
      },
      {
        path: "engineer",
        element: (
          <RequirePermission permission="control_simulation">
            <EngineerScreen />
          </RequirePermission>
        ),
      },
      {
        path: "audit",
        element: (
          <RequirePermission permission="read_audit">
            <AuditScreen />
          </RequirePermission>
        ),
      },
      {
        path: "users",
        element: (
          <RequirePermission permission="manage_users">
            <UsersScreen />
          </RequirePermission>
        ),
      },
      { path: "platform", element: <PlatformScreen /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
];

function SignedIn() {
  const [router] = useState(() => createBrowserRouter(routes));
  return (
    <LiveProvider>
      <RouterProvider router={router} />
    </LiveProvider>
  );
}

function SessionGate() {
  const { state } = useSession();
  switch (state.status) {
    case "restoring":
      return (
        <p role="status" className="login muted">
          Restoring your session…
        </p>
      );
    case "signed_out":
      return <LoginScreen notice={state.notice} />;
    case "signed_in":
      return <SignedIn />;
  }
}

export function App() {
  return (
    <SessionProvider>
      <SessionGate />
    </SessionProvider>
  );
}
