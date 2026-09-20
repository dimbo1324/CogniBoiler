// The place the deferred AI stage will fill. VISION §11 reserves the names it will use —
// REST /api/v1/insights, gRPC InsightService, MQTT insights/* — and asks the console to
// keep a slot for its recommendations, hidden while no such service runs. Nothing in the
// console may depend on it: with VITE_INSIGHTS unset, this renders nothing at all.

import { EmptyNote } from "../components/ui/Note";
import { InfoIcon } from "../components/ui/icons";
import { Panel } from "../components/ui/Panel";

export function insightsEnabled(): boolean {
  return import.meta.env.VITE_INSIGHTS === "true";
}

export function RecommendationsPanel() {
  if (!insightsEnabled()) {
    return null;
  }
  return (
    <Panel title="Recommendations" glyph={InfoIcon}>
      <EmptyNote status>No recommendations: the insight service does not answer yet.</EmptyNote>
    </Panel>
  );
}
