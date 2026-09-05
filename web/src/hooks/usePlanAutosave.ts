import { useRef, useState } from "react";
import { api, Plan } from "../api";
import { PLAN_SAVE_DEBOUNCE_MS } from "../constants";
import { SaveState } from "../types";

/**
 * Debounced, serialized autosave for plan.json.
 *
 * The semantics here are subtle enough to be worth isolating. Edits coalesce
 * into one trailing PUT so rapid keystrokes cannot complete out of order and
 * let an older request overwrite a newer edit on disk. Saves are serialized
 * through a promise chain that always resolves, so one failure does not
 * permanently break every save after it -- while the *real* outcome of the
 * most recent save is tracked separately, because `flush()` must throw when a
 * save actually failed. Generating against a plan that failed to save would
 * silently use stale on-disk content.
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

  // saveChainRef is a promise chain used purely to serialize save
  // *execution order* -- it always resolves, even after a failed save, so
  // one failure doesn't permanently break every save after it.
  // lastSaveOutcomeRef instead reflects the *real* outcome of the most
  // recently started save; flushPendingPlanSave() awaits that (not the
  // chain) so a failed save is never silently treated as success --
  // generate() must not enqueue a job against a plan that failed to save.
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingSaveRef = useRef<{ projectId: string; plan: Plan } | null>(null);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());
  const lastSaveOutcomeRef = useRef<Promise<void>>(Promise.resolve());

  function enqueueSave(): Promise<void> {
    const runSave = saveChainRef.current.then(async () => {
      const pending = pendingSaveRef.current;
      if (!pending) return;
      // Take ownership of this pending value before the request starts (not
      // after) so a newer edit made while the request is in flight lands in
      // a fresh slot instead of being clobbered when this save resolves.
      pendingSaveRef.current = null;
      try {
        setSaveState("saving");
        await api.savePlan(pending.projectId, pending.plan);
        if (pendingSaveRef.current === null) setSaveState("saved");
      } catch (err) {
        // Put the failed edit back so a later flush can retry it -- but
        // only if nothing newer has already claimed the slot, otherwise
        // this would overwrite (and lose) that newer edit.
        if (pendingSaveRef.current === null) {
          pendingSaveRef.current = pending;
        }
        setSaveState("error");
        throw err;
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
    const plan = { ...context.plan, [key]: value };
    onPlanChange(plan);

    pendingSaveRef.current = { projectId: context.projectId, plan };
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

  /** Drop any pending edit without saving it -- for switching away from a
   *  project whose edits have already been flushed, or discarded. */
  function reset(): void {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    pendingSaveRef.current = null;
  }

  return { saveState, setSaveState, savePlanField, flushPendingPlanSave, reset };
}
