import { useEffect, useEffectEvent, useId, useRef, type ReactNode } from "react";

const FOCUSABLE =
  'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])';

/** Tab and Shift+Tab cycle inside the dialog, so no control behind it can be reached. */
function trapTab(event: KeyboardEvent, dialog: HTMLElement): void {
  const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE));
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement;
  if (first === undefined || last === undefined) {
    event.preventDefault();
    return;
  }
  const inside = active instanceof Node && dialog.contains(active);
  if (event.shiftKey && (!inside || active === first)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (!inside || active === last)) {
    event.preventDefault();
    first.focus();
  }
}

/**
 * Every command that reaches the plant asks first. The dialog says what will change;
 * Escape or the backdrop cancels, except while the command is on its way.
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
  const dialog = useRef<HTMLDivElement>(null);
  const confirmButton = useRef<HTMLButtonElement>(null);
  // Live data re-renders the screen twice a second; the key listener and the initial focus
  // are set up once and always read the latest props.
  const cancel = useEffectEvent(() => {
    if (!busy) {
      onCancel();
    }
  });

  useEffect(() => {
    confirmButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cancel();
      } else if (event.key === "Tab" && dialog.current !== null) {
        trapTab(event, dialog.current);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <div
      className="modal-backdrop"
      onClick={() => {
        if (!busy) {
          onCancel();
        }
      }}
    >
      <div
        ref={dialog}
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
