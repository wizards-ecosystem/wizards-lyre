// What the next Generate will do, for labels only (SPEC.md sec 16.3). The
// server's resolver (server/generation/brief.py) decides; this mirrors its
// mode rule so the operator can predict the result before clicking, and
// never decides precedence itself.
import type { FieldOrigin, Plan, TrackedField } from "../api";

export type GenerateIntent = "plan" | "another-take" | "render-brief" | "free";

const INSTRUMENTAL_LYRICS = "[instrumental]";

export function isInstrumentalPlaceholder(lyrics: string): boolean {
  return lyrics.trim().toLowerCase() === INSTRUMENTAL_LYRICS;
}

/** Instrumental is on but the plan holds words: the server refuses this
 *  (`instrumental_lyrics_conflict`) until the user resolves it. */
export function hasInstrumentalConflict(plan: Plan): boolean {
  return plan.instrumental && plan.lyrics.trim() !== "" && !isInstrumentalPlaceholder(plan.lyrics);
}

export function originOf(plan: Plan, field: TrackedField): FieldOrigin {
  return plan.field_origins?.[field] ?? "unknown";
}

function planMode(plan: Plan): "from_idea" | "from_brief" {
  if (plan.creation_mode) return plan.creation_mode;
  return plan.query.trim() && !plan.caption && !plan.lyrics ? "from_idea" : "from_brief";
}

export function generateIntent(plan: Plan): GenerateIntent {
  const hasIdea = plan.query.trim() !== "";
  const hasBrief = plan.caption.trim() !== "" || plan.lyrics.trim() !== "";
  let mode = planMode(plan);
  if (mode === "from_idea" && !hasIdea) mode = "from_brief";
  if (mode === "from_brief" && !hasBrief && hasIdea) mode = "from_idea";
  if (mode === "from_idea") return "plan";
  if (!hasBrief) return "free";
  const planned = Object.values(plan.field_origins ?? {}).includes("planner");
  return planned ? "another-take" : "render-brief";
}

export const INTENT_COPY: Record<GenerateIntent, { title: string; detail: string }> = {
  plan: {
    title: "Generate plans from your idea",
    detail:
      "The planner writes the caption and lyrics you have not written, and any musical field you have not set. Your own text and settings are kept.",
  },
  "another-take": {
    title: "Generate makes another take of this brief",
    detail:
      "Same caption, lyrics, and settings, with a new seed. Re-plan to let the planner start again from the idea.",
  },
  "render-brief": {
    title: "Generate renders your brief as written",
    detail: "Caption, lyrics, and the fields you set go to ACE-Step; unset fields are its choice.",
  },
  free: {
    title: "Generate lets ACE-Step choose everything",
    detail: "Describe an idea, or write a caption or lyrics, to steer it.",
  },
};

/** Short badge text and an explanation for a field's origin. */
export function originBadge(
  origin: FieldOrigin,
  hasValue: boolean,
): { label: string; title: string } | null {
  switch (origin) {
    case "user":
      return { label: "yours", title: "You set this. Planning will not change it." };
    case "planner":
      return {
        label: "planned",
        title: "Filled by the planner from your idea. Edit it to make it yours.",
      };
    case "preset":
      return { label: "preset", title: "Set by a preset. Edit it to make it yours." };
    case "default":
      return hasValue
        ? { label: "default", title: "Application default, not a lock. The planner may change it." }
        : { label: "auto", title: "Not set: the planner or ACE-Step chooses." };
    default:
      return hasValue
        ? { label: "older plan", title: "Written before Lyre tracked who set each field." }
        : null;
  }
}
