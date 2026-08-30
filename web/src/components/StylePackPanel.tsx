import { Job, Lora, Take } from "../api";
import { MIN_LORA_SOURCE_TAKES } from "../constants";
import { StylePackPanel as StylePackShell } from "./Workbench";

/**
 * The style-pack (LoRA) training room: trained packs, any training still in
 * flight, and the control that starts a new one.
 *
 * Like the library, this owns a slice of state nothing else reads --
 * the selected source takes, the pack name, and the recovered training jobs
 * (SPEC.md sec 4.4).
 */
export function StylePackPanel({
  active,
  loras,
  trainingJobs,
  takes,
  loraSourceIds,
  onToggleSource,
  loraName,
  onNameChange,
  onTrain,
  busy,
  activeJobAction,
}: {
  active: boolean;
  loras: Lora[];
  trainingJobs: Job[];
  takes: Take[];
  loraSourceIds: Set<string>;
  onToggleSource: (takeId: string) => void;
  loraName: string;
  onNameChange: (value: string) => void;
  onTrain: () => void;
  busy: boolean;
  activeJobAction: string | null;
}) {
  return (
    <StylePackShell active={active}>
      <div className="style-heading">
        <div>
          <span className="eyebrow">Training room</span>
          <h3>Style packs</h3>
        </div>
        <p>Build a local style from eight or more successful takes.</p>
      </div>
      {loras.length === 0 && trainingJobs.length === 0 && (
        <p className="hint">No style packs trained yet.</p>
      )}
      <ul className="lora-list">
        {trainingJobs.map((job) => (
          <li key={job.id} className="lora-training">
            <span className="lora-name">Training style pack…</span>
            <span className="lora-status">
              {job.status === "queued" ? "queued — waiting for the GPU" : "running"}
            </span>
          </li>
        ))}
        {loras.map((lora) => (
          <li key={lora.id} className={lora.error ? "lora-error" : ""}>
            <span className="lora-name">{lora.name}</span>
            <span className="lora-status">
              {lora.error ? `error: ${lora.error}` : (lora.status ?? "—")}
            </span>
            <span className="lora-loss">
              {lora.final_loss != null ? `loss ${lora.final_loss.toFixed(4)}` : ""}
            </span>
          </li>
        ))}
      </ul>
      <div className="style-source-heading">
        <strong>Training sources</strong>
        <span>
          {loraSourceIds.size}/{MIN_LORA_SOURCE_TAKES}
        </span>
      </div>
      <ul className="style-source-list">
        {takes.map((take) => (
          <li key={take.id}>
            <label>
              <input
                type="checkbox"
                className="lora-source-checkbox"
                title="Include in style pack training source"
                checked={loraSourceIds.has(take.id)}
                onChange={() => onToggleSource(take.id)}
              />
              <span>
                <strong>{take.task_type}</strong>
                <small>
                  seed {take.seed} ·{" "}
                  {take.duration_sec != null ? `${take.duration_sec.toFixed(1)}s` : "—"}
                </small>
              </span>
            </label>
          </li>
        ))}
      </ul>
      <div className="lora-train-panel">
        <label>
          Style pack name
          <input
            placeholder="my-style"
            value={loraName}
            onChange={(event) => onNameChange(event.target.value)}
          />
        </label>
        <button
          type="button"
          onClick={onTrain}
          disabled={
            busy ||
            trainingJobs.length > 0 ||
            loraSourceIds.size < MIN_LORA_SOURCE_TAKES ||
            !loraName.trim()
          }
          title={
            trainingJobs.length > 0
              ? "A style pack is already training for this project"
              : loraSourceIds.size < MIN_LORA_SOURCE_TAKES
                ? `Select at least ${MIN_LORA_SOURCE_TAKES} takes first`
                : !loraName.trim()
                  ? "Enter a style pack name first"
                  : undefined
          }
        >
          {busy && activeJobAction === "style pack training"
            ? "Training…"
            : trainingJobs.length > 0
              ? `Training… (${trainingJobs[0].status})`
              : "Train style pack"}
        </button>
      </div>
    </StylePackShell>
  );
}
