import { useEffect, useRef } from "react";
import { ConfirmationRequest } from "../types";

export function ConfirmationDialog({
  request,
  onDecision,
}: {
  request: ConfirmationRequest | null;
  onDecision: (accepted: boolean) => void;
}) {
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!request) return;
    cancelRef.current?.focus();

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onDecision(false);
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled])"),
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [request, onDecision]);

  if (!request) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={() => onDecision(false)}>
      <div
        ref={dialogRef}
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-message"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <span className="eyebrow">Confirm action</span>
        <h2 id="confirm-title">{request.title}</h2>
        <p id="confirm-message">{request.message}</p>
        <div className="dialog-actions">
          <button ref={cancelRef} type="button" className="button-secondary" onClick={() => onDecision(false)}>
            Cancel
          </button>
          <button
            type="button"
            className={request.destructive ? "button-danger" : "button-primary"}
            onClick={() => onDecision(true)}
          >
            {request.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
