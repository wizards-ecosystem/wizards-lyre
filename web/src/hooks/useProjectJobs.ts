import { useCallback, useEffect, useRef, useState } from "react";
import { api, Job, TERMINAL_JOB_STATUSES } from "../api";
import {
  JOB_DISCOVERY_EVERY_TICKS,
  JOB_IDLE_DISCOVERY_MS,
  JOB_POLL_INTERVAL_MS,
  JOB_SLOW_POLL_AFTER_MS,
  JOB_SLOW_POLL_INTERVAL_MS,
} from "../constants";
import { resumePendingSubmission } from "../lib/jobs";

/**
 * Observes every job of the open project (SPEC.md sec 16.5/16.6).
 *
 * The queue, not component memory, decides what happened. On open, reload,
 * or reconnect this lists the project's queued/running jobs -- every
 * action, not only style-pack training -- and attaches to them, then polls
 * each one until it is terminal (`done`, `error`, or `cancelled`). A job
 * started by another tab or an agent is discovered on a periodic re-list.
 *
 * Observation never gives up: past JOB_SLOW_POLL_AFTER_MS it only polls
 * less often, because a slow render is not a failed one. A failed poll is a
 * lost observation, not a lost job; it is retried on the next tick.
 *
 * Polling follows the project id, so switching projects stops observing
 * the old project's jobs, and a response that lands after a switch is
 * dropped. `onFinished` fires once per job, only for the project on screen.
 */
export function useProjectJobs({
  projectId,
  onFinished,
}: {
  projectId: string | null;
  onFinished: (job: Job) => void;
}) {
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [lastFinished, setLastFinished] = useState<Job | null>(null);
  const projectRef = useRef<string | null>(projectId);
  const activeRef = useRef<Map<string, { job: Job; since: number }>>(new Map());
  const finishedIds = useRef<Set<string>>(new Set());
  // The caller's handler changes every render; the latest one is read when
  // a job finishes, without re-subscribing the poll loop.
  const onFinishedRef = useRef(onFinished);
  useEffect(() => {
    onFinishedRef.current = onFinished;
  });

  const publish = useCallback(() => {
    setActiveJobs(Array.from(activeRef.current.values(), (entry) => entry.job));
  }, []);

  const observe = useCallback(
    (job: Job, opts: { collect?: boolean } = {}) => {
      if (job.project_id !== projectRef.current) return;
      if (TERMINAL_JOB_STATUSES.includes(job.status)) {
        if (finishedIds.current.has(job.id)) return;
        finishedIds.current.add(job.id);
        const wasActive = activeRef.current.delete(job.id);
        publish();
        setLastFinished(job);
        // A job this page tracked is always collected, even when it was
        // already finished by the time it was tracked (a resend after a
        // lost response can return a job that is done).
        if (wasActive || opts.collect) onFinishedRef.current(job);
        return;
      }
      const existing = activeRef.current.get(job.id);
      activeRef.current.set(job.id, { job, since: existing?.since ?? Date.now() });
      publish();
    },
    [publish],
  );

  // Start observing a job this page just submitted, and look at it on the
  // fast cadence right away rather than after an idle wait.
  const kickRef = useRef<() => void>(() => {});
  const track = useCallback(
    (job: Job) => {
      observe(job, { collect: true });
      kickRef.current();
    },
    [observe],
  );

  useEffect(() => {
    projectRef.current = projectId;
    activeRef.current = new Map();
    finishedIds.current = new Set();
    // Deliberate: a project switch clears the previous project's job view
    // in the same pass as it starts observing the new one.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setActiveJobs([]);
    setLastFinished(null);
    if (!projectId) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let ticks = 0;

    async function discover() {
      try {
        const jobs = await api.listJobs({ projectId: projectId!, active: true });
        if (cancelled || projectRef.current !== projectId) return;
        for (const job of jobs) {
          if (!TERMINAL_JOB_STATUSES.includes(job.status)) observe(job);
        }
      } catch {
        // The server may be restarting; the next discovery tries again.
      }
    }

    async function tick() {
      ticks += 1;
      if (ticks % JOB_DISCOVERY_EVERY_TICKS === 0) await discover();
      for (const { job } of Array.from(activeRef.current.values())) {
        try {
          const latest = await api.getJob(job.id);
          if (cancelled || projectRef.current !== projectId) return;
          observe(latest);
        } catch {
          // A lost observation; keep the job and look again next tick.
        }
      }
      if (cancelled) return;
      schedule();
    }

    async function idleTick() {
      await discover();
      if (!cancelled) schedule();
    }

    function schedule() {
      if (activeRef.current.size === 0) {
        // Nothing to observe: only look for work another client started.
        timer = setTimeout(idleTick, JOB_IDLE_DISCOVERY_MS);
        return;
      }
      const oldest = Math.min(...Array.from(activeRef.current.values(), (entry) => entry.since));
      const slow = Date.now() - oldest > JOB_SLOW_POLL_AFTER_MS;
      timer = setTimeout(tick, slow ? JOB_SLOW_POLL_INTERVAL_MS : JOB_POLL_INTERVAL_MS);
    }

    kickRef.current = () => {
      if (cancelled) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(tick, 0);
    };

    (async () => {
      const resumed = await resumePendingSubmission(projectId);
      if (cancelled) return;
      if (resumed) observe(resumed, { collect: true });
      await discover();
      if (!cancelled) schedule();
    })();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [projectId, observe]);

  return { activeJobs, lastFinished, track };
}
