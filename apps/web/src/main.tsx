import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "uplot/dist/uPlot.min.css";
import "./styles.css";

import { App } from "./App";
import { ApiError } from "./api/http";
import { applyTheme, storedTheme } from "./theme";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html has no #root element to mount the console into");
}

applyTheme(storedTheme());

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // A refused request (4xx) will be refused again; only the network and upstream
      // outages are worth a retry.
      retry: (failures, error) =>
        failures < 2 && !(error instanceof ApiError && error.status >= 400 && error.status < 500),
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
