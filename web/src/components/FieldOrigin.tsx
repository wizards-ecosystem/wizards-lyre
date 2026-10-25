import type { Plan, TrackedField } from "../api";
import { originBadge, originOf } from "../lib/creation";

/** Who set a plan field (SPEC.md sec 16.3): yours, planned, preset, default,
 *  or auto. Shown beside the field so explicit locks and inferred values
 *  are never mistaken for each other. Hidden from the accessible name of the
 *  field it sits in; the Generate summary lists the locked fields in text. */
export function FieldOrigin({ plan, field }: { plan: Plan; field: TrackedField }) {
  const value = plan[field];
  const hasValue = value !== null && value !== undefined && String(value).trim() !== "";
  const badge = originBadge(originOf(plan, field), hasValue);
  if (!badge) return null;
  return (
    <span
      className={`field-origin origin-${originOf(plan, field)}`}
      title={badge.title}
      aria-hidden="true"
    >
      {badge.label}
    </span>
  );
}
