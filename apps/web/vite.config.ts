import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server proxies every gateway route, so the browser always talks to one origin —
// the same shape the production nginx container will have.
const gateway = "http://localhost:8000";

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
