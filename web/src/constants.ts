// Tunables shared across the studio UI. Each one that mirrors a server-side
// value says so, so the two can be kept in step.

export const HEALTH_POLL_INTERVAL_MS = 5000;
export const JOB_POLL_INTERVAL_MS = 1000;
// Observation never times out (SPEC.md sec 16.5: a slow render is not a
// failed one). Past this it only polls less often; style-pack training is
// hour-scale and lands here too.
export const JOB_SLOW_POLL_AFTER_MS = 10 * 60 * 1000;
export const JOB_SLOW_POLL_INTERVAL_MS = 5000;
// While jobs are active, re-list the project's queue every few ticks to
// pick up work another tab or an agent started.
export const JOB_DISCOVERY_EVERY_TICKS = 5;
// With nothing active, that re-list is all the polling there is.
export const JOB_IDLE_DISCOVERY_MS = 5000;
// A submission whose response was lost is resent under the same identity
// this many times before the page gives up and keeps it for a reload.
export const SUBMIT_RETRY_ATTEMPTS = 3;
// A remembered submission older than this is not resent on reload.
export const PENDING_SUBMISSION_MAX_AGE_MS = 10 * 60 * 1000;

// SPEC.md sec 4.4 "Style pack | LoRA train / load | 8+ songs" -- mirrors
// server.jobs.MIN_LORA_SOURCE_TAKES so the Train button can disable itself
// before even attempting a request the server would reject.
export const MIN_LORA_SOURCE_TAKES = 8;

// Coalesce rapid keystrokes into one PUT instead of firing one per
// keystroke (which can complete out of order and let an older request
// overwrite a newer edit on disk).
export const PLAN_SAVE_DEBOUNCE_MS = 500;
// A failed plan save is retried on this schedule (the last delay repeats)
// while the edit stays on screen and in the local draft (SPEC.md sec 16.6).
export const PLAN_SAVE_RETRY_DELAYS_MS = [2000, 5000, 15000, 30000] as const;

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
