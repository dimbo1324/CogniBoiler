import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server proxies every gateway route, so the browser always talks to one origin —
// the same shape the production nginx container will have.
const gateway = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
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
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
