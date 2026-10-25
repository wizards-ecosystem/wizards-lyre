// Job bodies for POST /api/projects/{id}/jobs (SPEC.md sec 8.1), one builder
// per studio action. They only build the body; api.enqueue sends it with
// the submission identity, and useProjectJobs observes the job.
import type { JobRequest } from "../api";

type Source = { takeId: string } | { uploadPath: string };

function sourceFields(source: Source): Record<string, string> {
  // Exactly one of source_take_id/upload_path is ever sent, matching
  // server.jobs._resolve_source_audio's "source_take_id or upload_path"
  // contract -- callers pass whichever source is active.
  return "takeId" in source
    ? { source_take_id: source.takeId }
    : { upload_path: source.uploadPath };
}

function lora(loraId?: string | null): Record<string, string> {
  // SPEC.md sec 4.4 "LoRA train / load": with a pack selected the server
  // coerces the omitted dit_profile to studio_ops, so callers must not
  // also pass an explicit profile or the server rejects the conflict.
  return loraId ? { lora_id: loraId } : {};
}

export const jobBodies = {
  // dit_profile is intentionally omitted: the server falls back to the
  // project's own persisted dit_profile (PATCH /api/projects/{id}).
  //
  // seed (SPEC.md sec 7.3): -1 means the worker picks a seed and records the
  // actual one used in the take meta; any other integer is a fixed seed.
  //
  // creationMode (SPEC.md sec 16.3) is sent only to override the plan's own
  // mode, e.g. "from_idea" to re-plan an unchanged idea.
  generate: (
    loraId?: string | null,
    seed: number = -1,
    creationMode?: "from_idea" | "from_brief",
  ): JobRequest => ({
    action: "generate",
    seed,
    ...lora(loraId),
    ...(creationMode ? { creation_mode: creationMode } : {}),
  }),
  cover: (
    source: Source,
    strength: number,
    loraId?: string | null,
    seed: number = -1,
  ): JobRequest => ({
    action: "cover",
    ...sourceFields(source),
    audio_cover_strength: strength,
    seed,
    ...lora(loraId),
  }),
  repaint: (
    source: Source,
    start: number,
    end: number,
    loraId?: string | null,
    seed: number = -1,
  ): JobRequest => ({
    action: "repaint",
    ...sourceFields(source),
    repainting_start: start,
    repainting_end: end,
    seed,
    ...lora(loraId),
  }),
  extract: (sourceTakeId: string, trackName: string): JobRequest => ({
    action: "extract",
    dit_profile: "studio_ops",
    source_take_id: sourceTakeId,
    track_name: trackName,
    seed: -1,
  }),
  lego: (
    sourceTakeId: string,
    trackName: string,
    region: { start: number; end: number } | null,
  ): JobRequest => ({
    action: "lego",
    dit_profile: "studio_ops",
    source_take_id: sourceTakeId,
    track_name: trackName,
    seed: -1,
    ...(region ? { repainting_start: region.start, repainting_end: region.end } : {}),
  }),
  complete: (sourceTakeId: string, trackName: string): JobRequest => ({
    action: "complete",
    dit_profile: "studio_ops",
    source_take_id: sourceTakeId,
    track_name: trackName,
    seed: -1,
  }),
  // source_take_ids must contain 8+ distinct take ids -- server.jobs
  // enforces MIN_LORA_SOURCE_TAKES (SPEC.md sec 4.4 "8+ songs").
  trainLora: (sourceTakeIds: string[], name: string): JobRequest => ({
    action: "train_lora",
    source_take_ids: sourceTakeIds,
    name,
    seed: -1,
  }),
};
