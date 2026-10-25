import { useRef, useState } from "react";
import { api, ApiError, Plan } from "../api";
import { PLAN_SAVE_DEBOUNCE_MS, PLAN_SAVE_RETRY_DELAYS_MS } from "../constants";
import { SaveState } from "../types";

/** The newest unsent edit for a project, kept in localStorage so a reload,
 *  crash, or failed save does not lose it (SPEC.md sec 16.6). `baseRevision`
 *  is the server revision the edit started from, so a stale draft is never
 *  silently written over a newer plan. */
export interface LocalDraft {
  projectId: string;
  plan: Plan;
  baseRevision: number | null;
  savedAt: number;
}

const draftKey = (projectId: string) => `lyre.planDraft.${projectId}`;

export function readLocalDraft(projectId: string): LocalDraft | null {
  try {
    const raw = localStorage.getItem(draftKey(projectId));
    return raw ? (JSON.parse(raw) as LocalDraft) : null;
  } catch {
    return null;
  }
}

export function clearLocalDraft(projectId: string): void {
  try {
    localStorage.removeItem(draftKey(projectId));
  } catch {
    // ignore: storage is a convenience
  }
}

function writeLocalDraft(draft: LocalDraft): void {
  try {
    localStorage.setItem(draftKey(draft.projectId), JSON.stringify(draft));
  } catch {
    // ignore: storage is a convenience
  }
}

/**
 * Debounced, serialized, revision-checked autosave for plan.json.
 *
 * Edits coalesce into one trailing PUT so rapid keystrokes cannot complete
 * out of order and let an older request overwrite a newer edit on disk.
 * Saves are serialized through a promise chain that always resolves, so one
 * failure does not permanently break every save after it -- while the
 * *real* outcome of the most recent save is tracked separately, because
 * `flush()` must throw when a save actually failed. Generating against a
 * plan that failed to save would silently use stale on-disk content.
 *
 * Every save names the server revision it was based on (`expected_revision`,
 * SPEC.md sec 16.3). A plan that changed elsewhere answers 409: the edit is
 * kept, nothing is overwritten, and the state becomes `conflict` until the
 * user keeps their edit or takes the saved plan. Any other failure is
 * retried on PLAN_SAVE_RETRY_DELAYS_MS while the text stays on screen and
 * in the local draft.
 *
 * There is one pending slot, shared across projects, so switching projects
 * must `flush()` first or the outgoing project's edit is discarded by the
 * incoming project's.
 */
export function usePlanAutosave({
  getContext,
  onPlanChange,
  onError,
}: {
  /** The open project and its current plan, or null when none is open. Read
   *  at call time rather than captured, so this never holds a stale plan. */
  getContext: () => { projectId: string; plan: Plan } | null;
  /** Applies the optimistic local update; the caller owns the plan state. */
  onPlanChange: (plan: Plan) => void;
  onError: (message: string) => void;
}) {
  const [saveState, setSaveState] = useState<SaveState>("idle");
  // Whether a failed save will be retried on its own. Only a lost response
  // or a server error is: a rejected edit (4xx) would fail the same way
  // again, so it waits for the user (SPEC.md sec 16.5: no blind retries).
  const [retrying, setRetrying] = useState(false);

  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryAttemptRef = useRef(0);
  const pendingSaveRef = useRef<{ projectId: string; plan: Plan } | null>(null);
  const inFlightRef = useRef(0);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());
  const lastSaveOutcomeRef = useRef<Promise<void>>(Promise.resolve());
  // The plan revision the server last reported for the open project.
  const revisionRef = useRef<{ projectId: string; revision: number | null } | null>(null);

  function baseRevision(projectId: string): number | null {
    return revisionRef.current?.projectId === projectId ? revisionRef.current.revision : null;
  }

  /** Called whenever a plan arrives from the server (load, refresh, save). */
  function setServerRevision(projectId: string, revision: number | null | undefined): void {
    revisionRef.current = { projectId, revision: revision ?? null };
  }

  function cancelRetry(): void {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }

  function scheduleRetry(): void {
    cancelRetry();
    const delays = PLAN_SAVE_RETRY_DELAYS_MS;
    const delay = delays[Math.min(retryAttemptRef.current, delays.length - 1)];
    retryAttemptRef.current += 1;
    retryTimeoutRef.current = setTimeout(() => {
      retryTimeoutRef.current = null;
      enqueueSave().catch(() => {});
    }, delay);
  }

  function enqueueSave(opts: { force?: boolean } = {}): Promise<void> {
    const runSave = saveChainRef.current.then(async () => {
      const pending = pendingSaveRef.current;
      if (!pending) return;
      // Take ownership of this pending value before the request starts (not
      // after) so a newer edit made while the request is in flight lands in
      // a fresh slot instead of being clobbered when this save resolves.
      pendingSaveRef.current = null;
      inFlightRef.current += 1;
      try {
        setSaveState("saving");
        const saved = await api.savePlan(pending.projectId, pending.plan, {
          expectedRevision: opts.force ? null : baseRevision(pending.projectId),
        });
        setServerRevision(pending.projectId, saved.revision);
        retryAttemptRef.current = 0;
        setRetrying(false);
        // Re-read: an edit may have claimed the slot while this was in flight
        // (TypeScript still sees the null assigned above).
        const newer = pendingSaveRef.current as { projectId: string; plan: Plan } | null;
        if (newer === null) {
          clearLocalDraft(pending.projectId);
          setSaveState("saved");
        } else {
          // A newer edit is waiting; keep its draft current with the new base.
          writeLocalDraft({
            ...newer,
            baseRevision: saved.revision ?? null,
            savedAt: Date.now(),
          });
        }
      } catch (err) {
        // Put the failed edit back so a later flush can retry it -- but
        // only if nothing newer has already claimed the slot, otherwise
        // this would overwrite (and lose) that newer edit.
        if (pendingSaveRef.current === null) {
          pendingSaveRef.current = pending;
        }
        if (err instanceof ApiError && err.status === 409) {
          setSaveState("conflict");
          setRetrying(false);
        } else if (err instanceof ApiError && err.status < 500) {
          setSaveState("error");
          setRetrying(false);
        } else {
          setSaveState("error");
          setRetrying(true);
          scheduleRetry();
        }
        throw err;
      } finally {
        inFlightRef.current -= 1;
      }
    });
    saveChainRef.current = runSave.catch(() => {});
    lastSaveOutcomeRef.current = runSave;
    return runSave;
  }

  // Cancels any pending debounce and waits for the latest edit (plus
  // anything already in flight) to finish saving. generate() and
  // switchActiveProject() must call this first: without it, a job can
  // start (or a project switch can happen) within the debounce window and
  // read/discard the plan from before the user's last edit. Throws if the
  // save actually failed, so callers can abort instead of proceeding
  // against a stale on-disk plan.
  async function flushPendingPlanSave(): Promise<void> {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    cancelRetry();
    if (pendingSaveRef.current) {
      await enqueueSave();
    } else {
      await lastSaveOutcomeRef.current;
    }
  }

  /** Apply one plan field optimistically and schedule the debounced save. */
  function savePlanField<K extends keyof Plan>(key: K, value: Plan[K]): void {
    const context = getContext();
    if (!context) return;
    savePlan({ ...context.plan, [key]: value });
  }

  /** Apply a whole edited plan (several fields at once, e.g. a preset or
   *  "use settings") and schedule the debounced save. */
  function savePlan(plan: Plan, projectId?: string): void {
    // An explicit project id is for callers outside the render that owns
    // the plan (restoring a local draft right after a project opens).
    const target = projectId ?? getContext()?.projectId;
    if (!target) return;
    onPlanChange(plan);

    pendingSaveRef.current = { projectId: target, plan };
    writeLocalDraft({
      projectId: target,
      plan,
      baseRevision: baseRevision(target),
      savedAt: Date.now(),
    });
    setSaveState("saving");
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => {
      saveTimeoutRef.current = null;
      // Fire-and-forget from here (nothing is awaiting this save yet), but
      // still surface a failure -- flushPendingPlanSave() picks up the
      // real outcome via lastSaveOutcomeRef if something awaits it later.
      enqueueSave().catch((err) => onError(String(err)));
    }, PLAN_SAVE_DEBOUNCE_MS);
  }

  /** Retry a failed save now instead of waiting for the next attempt. */
  function retryNow(): Promise<void> {
    cancelRetry();
    return enqueueSave();
  }

  /** Resolve a 409: keep this edit (write it over the newer plan, which the
   *  user has now chosen to do) or drop it for the saved plan. */
  async function resolveConflict(choice: "keep-mine" | "use-saved"): Promise<void> {
    cancelRetry();
    if (choice === "keep-mine") {
      await enqueueSave({ force: true });
      return;
    }
    const pending = pendingSaveRef.current;
    pendingSaveRef.current = null;
    if (pending) clearLocalDraft(pending.projectId);
    setSaveState("idle");
  }

  /** True while an edit is waiting, being retried, or in flight: a refresh
   *  must not replace the plan on screen with an older server copy. */
  function hasPendingEdit(): boolean {
    return (
      pendingSaveRef.current !== null || saveTimeoutRef.current !== null || inFlightRef.current > 0
    );
  }

  /** Page is going away: send the last debounced edit with keepalive so the
   *  browser completes it, and keep the local draft in case it does not. */
  function flushOnUnload(): void {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    const pending = pendingSaveRef.current;
    if (!pending) return;
    api
      .savePlan(pending.projectId, pending.plan, {
        expectedRevision: baseRevision(pending.projectId),
        keepalive: true,
      })
      .catch(() => {});
  }

  /** Drop any pending edit without saving it -- for switching away from a
   *  project whose edits have already been flushed, or discarded. */
  function reset(): void {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    cancelRetry();
    retryAttemptRef.current = 0;
    pendingSaveRef.current = null;
  }

  return {
    saveState,
    retrying,
    setSaveState,
    savePlanField,
    savePlan,
    flushPendingPlanSave,
    retryNow,
    resolveConflict,
    hasPendingEdit,
    flushOnUnload,
    setServerRevision,
    reset,
  };
}
