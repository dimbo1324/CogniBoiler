import { useMutation, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { CommandResult, type Acknowledgement } from "./CommandResult";
import { ConfirmDialog } from "./ConfirmDialog";
import { OkIcon, RefusedIcon } from "./ui/icons";
import { Panel } from "./ui/Panel";

/** A command the operator confirms before it runs. */
export interface PendingAction<T> {
  title: string;
  body: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  run: () => Promise<T>;
  /** What to say once it succeeded, where a screen says it in words. */
  done?: string;
}

export interface LastAction<T> {
  label: string;
  data: T | undefined;
  error: unknown;
}

export interface ConfirmedActionOptions<T> {
  /** Called when the operator confirms, before the command runs. */
  onConfirm?: (action: PendingAction<T>) => void;
  onSuccess?: (action: PendingAction<T>) => void;
}

/**
 * The one home of "confirm, run, show the last result": the pending action, the mutation,
 * the confirmation dialog and the invalidation of what the command changed.
 */
export function useConfirmedAction<T>(
  invalidate: QueryKey,
  options: ConfirmedActionOptions<T> = {},
) {
  const queryClient = useQueryClient();
  const [pending, setPending] = useState<PendingAction<T> | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: (run: () => Promise<T>) => run(),
    onSettled: () => queryClient.invalidateQueries({ queryKey: invalidate }),
  });

  const runNow = (actionLabel: string, run: () => Promise<T>) => {
    setLabel(actionLabel);
    mutation.mutate(run);
  };

  const dialog = pending && (
    <ConfirmDialog
      title={pending.title}
      confirmLabel={pending.confirmLabel}
      danger={pending.danger}
      busy={mutation.isPending}
      onCancel={() => {
        setPending(null);
      }}
      onConfirm={() => {
        const action = pending;
        setLabel(action.title);
        options.onConfirm?.(action);
        mutation.mutate(action.run, {
          onSuccess: () => {
            options.onSuccess?.(action);
          },
          onSettled: () => {
            setPending(null);
          },
        });
      }}
    >
      {pending.body}
    </ConfirmDialog>
  );

  const last: LastAction<T> | null =
    label === null ? null : { label, data: mutation.data, error: mutation.error };
  return { ask: setPending, runNow, dialog, last };
}

/** The answer to the last command, with a glyph that says whether it went through. */
export function LastActionPanel<T extends Acknowledgement>({
  title,
  last,
}: {
  title: string;
  last: LastAction<T> | null;
}) {
  if (last === null) {
    return null;
  }
  const refused = Boolean(last.error) || last.data?.accepted === false;
  return (
    <Panel
      title={title}
      glyph={refused ? RefusedIcon : OkIcon}
      headline={<span className="muted">: {last.label}</span>}
    >
      <CommandResult result={last.data} error={last.error} />
    </Panel>
  );
}
