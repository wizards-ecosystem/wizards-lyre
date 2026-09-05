// Tunables shared across the studio UI. Each one that mirrors a server-side
// value says so, so the two can be kept in step.

export const HEALTH_POLL_INTERVAL_MS = 5000;
export const JOB_POLL_INTERVAL_MS = 1000;
// Real ACE-Step generation can take a while (SPEC.md sec 5: it queues
// behind a dedicated worker process); give it a generous ceiling before
// giving up on polling rather than declaring failure too early.
export const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000;
// train_lora runs the whole training loop under the worker's GPU-exclusive
// lock and can take roughly an hour (SPEC.md sec 4.4) -- far longer than the
// ceiling above that's tuned for ordinary generate/cover/repaint jobs.
export const LORA_TRAIN_POLL_TIMEOUT_MS = 90 * 60 * 1000;
// Cadence for the recovery poll that watches a train_lora job discovered via
// GET /api/jobs (i.e. one that outlived a page refresh, or is running in
// another tab). Training itself is hour-scale, so this only needs to be
// prompt enough that completion shows up shortly after it happens.
export const LORA_TRAIN_RECOVERY_POLL_MS = 3000;

// SPEC.md sec 4.4 "Style pack | LoRA train / load | 8+ songs" -- mirrors
// server.jobs.MIN_LORA_SOURCE_TAKES so the Train button can disable itself
// before even attempting a request the server would reject.
export const MIN_LORA_SOURCE_TAKES = 8;

// Coalesce rapid keystrokes into one PUT instead of firing one per
// keystroke (which can complete out of order and let an older request
// overwrite a newer edit on disk).
export const PLAN_SAVE_DEBOUNCE_MS = 500;

// Identities for the two kinds of waveform regions (SPEC.md sec 9.2), which
// share one RegionsPlugin instance: the single ad-hoc repaint selection and
// the persisted plan section labels. The fixed id scheme is what lets each
// side tell its own regions apart -- creating a repaint selection must not
// clear the saved section labels, and re-rendering the labels must not touch
// the selection.
export const REPAINT_REGION_ID = "repaint-selection";
export const SECTION_REGION_ID_PREFIX = "section-label-";

// SPEC.md sec 4.1: the DiT profiles a generate/cover/repaint job may run
// under. studio_ops is deliberately not offered -- it is reserved for
// extract/lego/complete, or generate/cover/repaint with a style pack
// attached (server.jobs._resolve_dit_profile rejects/forces it either way).
export const DIT_PROFILE_OPTIONS = ["iterate", "polish", "quality"] as const;

// SPEC.md sec 4.4/9.2: "Lyrics carry structure tags such as [Verse],
// [Chorus], [Bridge], [Intro], [Outro]" -- the lyrics textarea is a "textarea
// with structure tags", so the palette below offers exactly this fixed set.
export const STRUCTURE_TAGS = ["Intro", "Verse", "Chorus", "Bridge", "Outro"] as const;
