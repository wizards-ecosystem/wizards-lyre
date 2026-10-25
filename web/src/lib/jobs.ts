import { api, ApiError, Job, JobRequest } from "../api";
import { PENDING_SUBMISSION_MAX_AGE_MS, SUBMIT_RETRY_ATTEMPTS } from "../constants";

// Submitting, observing, and collecting a job are separate steps (SPEC.md
// sec 16.5). This module owns the first: every submission carries an
// identity created before it is sent, so resending after a lost response
// returns the job the first send created instead of queuing a duplicate.
// Observation lives in hooks/useProjectJobs.ts.

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function newSubmissionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `sub-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

interface PendingSubmission {
  projectId: string;
  submissionId: string;
  body: JobRequest;
  createdAt: number;
}

const pendingKey = (projectId: string) => `lyre.pendingSubmission.${projectId}`;

// localStorage is a convenience here, not the source of truth: the queue is.
// Any storage failure (private mode, quota) just means a lost response can
// only be recovered by the in-page retry, not across a reload.
function savePending(pending: PendingSubmission): void {
  try {
    localStorage.setItem(pendingKey(pending.projectId), JSON.stringify(pending));
  } catch {
    // ignore
  }
}

function loadPending(projectId: string): PendingSubmission | null {
  try {
    const raw = localStorage.getItem(pendingKey(projectId));
    return raw ? (JSON.parse(raw) as PendingSubmission) : null;
  } catch {
    return null;
  }
}

function clearPending(projectId: string): void {
  try {
    localStorage.removeItem(pendingKey(projectId));
  } catch {
    // ignore
  }
}

// A fetch that never got an HTTP answer rejects with a TypeError. Anything
// the server did answer (ApiError) is final: 4xx means nothing was queued.
function isLostResponse(err: unknown): boolean {
  return !(err instanceof ApiError);
}

async function send(projectId: string, body: JobRequest, submissionId: string): Promise<Job> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await api.enqueue(projectId, body, submissionId);
    } catch (err) {
      if (!isLostResponse(err) || attempt + 1 >= SUBMIT_RETRY_ATTEMPTS) throw err;
      await sleep(250 * 2 ** attempt);
    }
  }
}

/** Queue one job. Resends the same submission id after a lost response;
 *  rejects with ApiError when the server refused it. A submission whose
 *  every send was lost is remembered for `resumePendingSubmission`. */
export async function submitJob(projectId: string, body: JobRequest): Promise<Job> {
  const pending: PendingSubmission = {
    projectId,
    submissionId: newSubmissionId(),
    body,
    createdAt: Date.now(),
  };
  savePending(pending);
  try {
    const job = await send(projectId, body, pending.submissionId);
    clearPending(projectId);
    return job;
  } catch (err) {
    if (!isLostResponse(err)) clearPending(projectId);
    throw err;
  }
}

/** On project open: a recent submission whose response never arrived is
 *  sent once more under its original identity. The server returns the job
 *  it already queued, or queues it now if the first send never landed. */
export async function resumePendingSubmission(projectId: string): Promise<Job | null> {
  const pending = loadPending(projectId);
  if (!pending) return null;
  if (Date.now() - pending.createdAt > PENDING_SUBMISSION_MAX_AGE_MS) {
    clearPending(projectId);
    return null;
  }
  try {
    const job = await api.enqueue(projectId, pending.body, pending.submissionId);
    clearPending(projectId);
    return job;
  } catch (err) {
    if (!isLostResponse(err)) clearPending(projectId);
    return null;
  }
}

/** Plain-language status for the job strip, from the job's real state. No
 *  percentages or ETAs: ACE-Step reports none (SPEC.md sec 16.5). */
export function describeJob(job: Job, opts: { loadingBaseModel?: boolean } = {}): string {
  const action = job.action === "train_lora" ? "style pack training" : job.action;
  if (job.status === "queued") {
    return job.stage === "needs_confirmation"
      ? `Needs confirmation: ${action}`
      : `Waiting for GPU: ${action}`;
  }
  if (job.status === "running") {
    if (job.cancel_requested) return `Cancelling: ${action}`;
    if (opts.loadingBaseModel) return `Loading base model: ${action}`;
    switch (job.stage) {
      case "loading_model":
        return `Loading model: ${action}`;
      case "planning":
        return `Planning the brief: ${action}`;
      case "rendering":
        return `Generating: ${action}`;
      case "writing_take":
        return `Writing the take: ${action}`;
      default:
        return `Running: ${action}`;
    }
  }
  if (job.status === "done") return `Complete: ${action}`;
  if (job.status === "cancelled") return `Cancelled: ${action}`;
  return `Interrupted: ${action}`;
}

/** A copyable one-paragraph summary of a failed job (SPEC.md sec 16.5). */
export function errorSummary(job: Job): string {
  return [
    `Lyre job ${job.id} (${job.action}, ${job.dit_profile}) ${job.status}`,
    job.stage ? `at stage ${job.stage}` : null,
    `project ${job.project_id}, updated ${job.updated_at}`,
    job.error ? `error: ${job.error}` : null,
  ]
    .filter(Boolean)
    .join("; ");
}
