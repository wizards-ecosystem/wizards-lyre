// Presentation helpers with no React or API dependency.

export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

// SPEC.md sec 7.3: "-1 from the user means worker picks and records" the
// seed. An empty input is the UI's way of saying -1; anything that doesn't
// parse to an integer also falls back to -1 rather than posting a value the
// server would reject as a validation error.
export function parseSeed(raw: string): number {
  const trimmed = raw.trim();
  if (trimmed === "") return -1;
  const parsed = Number(trimmed);
  return Number.isInteger(parsed) ? parsed : -1;
}
