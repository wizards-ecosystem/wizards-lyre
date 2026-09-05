import { useRef } from "react";
import { api } from "../api";
import { PLAN_SAVE_DEBOUNCE_MS } from "../constants";

/**
 * Debounced saves for a take's free-text notes (SPEC.md sec 12 Phase 6).
 *
 * The same coalescing and re-queue-on-failure semantics as
 * {@link usePlanAutosave}, but keyed by take id: several takes' notes fields
 * can each have an edit pending at once, unlike the plan's single shared slot.
 *
 * `flushAll` exists because a project switch or a page unload can happen while
 * more than one textarea has an unsaved edit, and `keepalive` is threaded
 * through for the unload case, where an ordinary fetch would be aborted
 * mid-flight by the navigation.
 */
export function useTakeNotes({
  getProjectId,
  onNotesChange,
  onError,
}: {
  /** Read at call time, not captured, so a debounce that fires after a project
   *  switch cannot save into the project the user just left. */
  getProjectId: () => string | null;
  /** Applies the optimistic local update; the caller owns the takes state. */
  onNotesChange: (takeId: string, notes: string) => void;
  onError: (message: string) => void;
}) {
  // Take notes debounce, analogous to the plan save mechanism above but
  // keyed by take id (several takes' notes fields can each have their own
  // edit in flight/pending at once, unlike the single shared plan slot).
  const takeSaveTimeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout> | null>>({});
  const pendingTakeNotesRef = useRef<Record<string, string>>({});

  // Sends whatever note edit is pending for `takeId` right now, canceling
  // its debounce timer first -- the single path every take-notes save
  // actually goes through, whether triggered by the debounce firing, a
  // blur, a project switch, or the page unloading. Re-queues the edit on
  // failure (mirroring enqueueSave's failure handling above) so a later
  // flush can retry it, but only if nothing newer has already claimed the
  // slot. `opts.keepalive` is passed straight through to `api.patchTake`
  // for the pagehide/beforeunload case, where a plain fetch would otherwise
  // be aborted mid-flight by the navigation.
  async function flushTakeNotes(takeId: string, opts?: { keepalive?: boolean }): Promise<void> {
    const timeout = takeSaveTimeoutsRef.current[takeId];
    if (timeout) {
      clearTimeout(timeout);
      takeSaveTimeoutsRef.current[takeId] = null;
    }
    if (!(takeId in pendingTakeNotesRef.current)) return;
    const projectId = getProjectId();
    const notes = pendingTakeNotesRef.current[takeId];
    delete pendingTakeNotesRef.current[takeId];
    if (!projectId) return;
    try {
      await api.patchTake(projectId, takeId, { notes }, opts);
    } catch (err) {
      if (!(takeId in pendingTakeNotesRef.current)) {
        pendingTakeNotesRef.current[takeId] = notes;
      }
      throw err;
    }
  }

  // Flushes every take's pending note edit (not just one) -- used before a
  // project switch, and from the pagehide/beforeunload handler below, since
  // either can happen while more than one take's textarea has an unsaved
  // edit in flight.
  async function flushAllPendingTakeNotes(opts?: { keepalive?: boolean }): Promise<void> {
    const takeIds = Object.keys(pendingTakeNotesRef.current);
    await Promise.all(takeIds.map((takeId) => flushTakeNotes(takeId, opts)));
  }

  function saveTakeNotes(takeId: string, notes: string): void {
    if (!getProjectId()) return;
    onNotesChange(takeId, notes);

    pendingTakeNotesRef.current[takeId] = notes;
    const existing = takeSaveTimeoutsRef.current[takeId];
    if (existing) clearTimeout(existing);
    takeSaveTimeoutsRef.current[takeId] = setTimeout(() => {
      flushTakeNotes(takeId).catch((err) => onError(String(err)));
    }, PLAN_SAVE_DEBOUNCE_MS);
  }

  /** Drop every pending note edit without saving -- for a project that has
   *  just been deleted, where a later save would 404. */
  function reset(): void {
    for (const timeout of Object.values(takeSaveTimeoutsRef.current)) {
      if (timeout) clearTimeout(timeout);
    }
    takeSaveTimeoutsRef.current = {};
    pendingTakeNotesRef.current = {};
  }

  return { saveTakeNotes, flushTakeNotes, flushAllPendingTakeNotes, reset };
}
