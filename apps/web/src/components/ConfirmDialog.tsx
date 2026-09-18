import { useEffect, useEffectEvent, useId, useRef, type ReactNode } from "react";

/**
 * Every command that reaches the plant asks first. The dialog says what will change;
 * Escape or the backdrop cancels.
 */
export function ConfirmDialog({
  title,
  children,
  confirmLabel,
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  title: string;
  children: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const titleId = useId();
  const confirmButton = useRef<HTMLButtonElement>(null);
  // Live data re-renders the screen twice a second; the key listener and the initial focus
  // are set up once and always call the latest handler.
  const cancel = useEffectEvent(onCancel);

  useEffect(() => {
    confirmButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        <h2 id={titleId}>{title}</h2>
        <div>{children}</div>
        <div className="row modal-actions">
          <button type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            ref={confirmButton}
            type="button"
            className={danger ? "danger" : "primary"}
            onClick={onConfirm}
            disabled={busy}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
