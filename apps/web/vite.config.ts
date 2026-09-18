import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server proxies every gateway route, so the browser always talks to one origin —
// the same shape the stack's nginx has. By default it goes through that nginx (the gateway
// has no host port); GATEWAY_URL points it at a gateway run from the host instead.
const gateway = process.env.GATEWAY_URL ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    // IPv4 loopback, like every published port of the stack: Node would otherwise bind
    // only ::1 for "localhost", which some browsers and sandboxes do not try.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": gateway,
      "/auth": gateway,
      "/health": gateway,
      "/ws": { target: gateway, ws: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
