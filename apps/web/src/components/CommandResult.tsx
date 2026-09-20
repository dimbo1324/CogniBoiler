import { Icon } from "./ui/Icon";
import { ErrorNote, ErrorOf } from "./ui/Note";
import { OkIcon, RefusedIcon } from "./ui/icons";

export interface Acknowledgement {
  accepted: boolean;
  reason: string;
}

/** What the PLC or the gateway answered to the last command. */
export function CommandResult({
  result,
  error,
}: {
  result: Acknowledgement | undefined;
  error: unknown;
}) {
  if (error) {
    return <ErrorOf error={error} />;
  }
  if (!result) {
    return null;
  }
  if (result.accepted) {
    return (
      <p className="note ok-text" role="status">
        <Icon glyph={OkIcon} tone="ok" />
        Accepted.
      </p>
    );
  }
  return <ErrorNote glyph={RefusedIcon}>Refused: {result.reason}</ErrorNote>;
}
