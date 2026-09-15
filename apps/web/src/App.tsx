import { useEffect, useState } from "react";

import { fetchHealth, type GatewayHealth } from "./api/client";

type GatewayState =
  | { kind: "checking" }
  | { kind: "reachable"; health: GatewayHealth }
  | { kind: "unreachable"; reason: string };

export function App() {
  const [gateway, setGateway] = useState<GatewayState>({ kind: "checking" });

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((health) => {
        setGateway({ kind: "reachable", health });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : String(error);
        setGateway({ kind: "unreachable", reason });
      });
    return () => {
      controller.abort();
    };
  }, []);

  return (
    <main>
      <h1>CogniBoiler operator console</h1>
      <p role="status">{describe(gateway)}</p>
    </main>
  );
}

function describe(gateway: GatewayState): string {
  switch (gateway.kind) {
    case "checking":
      return "Checking the API gateway…";
    case "reachable":
      return `API gateway ${gateway.health.status} (version ${gateway.health.version})`;
    case "unreachable":
      return `API gateway unreachable: ${gateway.reason}`;
  }
}
