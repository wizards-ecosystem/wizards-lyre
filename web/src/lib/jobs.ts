import { api, Job } from "../api";
import { JOB_POLL_INTERVAL_MS, JOB_POLL_TIMEOUT_MS } from "../constants";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// A job only finishes async, via server.jobs' queued -> running -> done|error
// lifecycle (SPEC.md sec 5) -- the enqueue response is just the initial
// `queued` row, so the caller has to keep polling /api/jobs/{id} itself.
export async function pollJob(
  jobId: string,
  onUpdate?: (job: Job) => void,
  timeoutMs: number = JOB_POLL_TIMEOUT_MS,
): Promise<Job> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await api.getJob(jobId);
    onUpdate?.(job);
    if (job.status === "done" || job.status === "error") {
      return job;
    }
    if (Date.now() > deadline) {
      throw new Error(`job ${jobId} is still ${job.status} after ${timeoutMs / 1000}s`);
    }
    await sleep(JOB_POLL_INTERVAL_MS);
  }
}
