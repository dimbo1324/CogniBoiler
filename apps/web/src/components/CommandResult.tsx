import { describeError } from "../api/http";

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
    return (
      <p className="error" role="alert">
        {describeError(error)}
      </p>
    );
  }
  if (!result) {
    return null;
  }
  return result.accepted ? (
    <p className="ok-text" role="status">
      Accepted.
    </p>
  ) : (
    <p className="error" role="alert">
      Refused: {result.reason}
    </p>
  );
}
