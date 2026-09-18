import { useState } from "react";
import { createBrowserRouter, Navigate, RouterProvider, type RouteObject } from "react-router";

import { Layout } from "./components/Layout";
import { LiveProvider } from "./live/LiveProvider";
import { AlarmsScreen } from "./screens/AlarmsScreen";
import { LoginScreen } from "./screens/LoginScreen";
import { OverviewScreen } from "./screens/OverviewScreen";
import { TrendsScreen } from "./screens/TrendsScreen";
import { SessionProvider, useSession } from "./session/SessionProvider";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <OverviewScreen /> },
      { path: "trends", element: <TrendsScreen /> },
      { path: "alarms", element: <AlarmsScreen /> },
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
