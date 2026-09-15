// The only module that talks to the network. Components call these functions and never
// `fetch` directly, so authentication and error handling have exactly one home.

export interface GatewayHealth {
  service: string;
  version: string;
  status: string;
}

export async function fetchHealth(signal?: AbortSignal): Promise<GatewayHealth> {
  const response = await fetch("/health", { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`health check failed with HTTP ${String(response.status)}`);
  }
  return (await response.json()) as GatewayHealth;
}
