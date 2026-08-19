# Wizard's Bard

Local generative music studio. ACE-Step 1.5 on the GPU. No cloud music APIs.

The product spec is **[SPEC.md](SPEC.md)**. Implement that file. Do not invent extra engines.

## Status

Bootstrap only. Conclave implements SPEC.md in phase order. There is no runnable studio yet.

## Machine

Windows, RTX 4070 Ti SUPER 16 GB. Default: ACE-Step 2B turbo + 1.7B LM. Bind `127.0.0.1:8421`.

## Layout

| Path | Role |
|---|---|
| `SPEC.md` | Sole product spec |
| `server/` | FastAPI (HTTP, jobs, files) |
| `worker/` | ACE-Step GPU process |
| `web/` | Vite + React studio |
| `tests/` | pytest, mocked worker, no GPU |
| `scripts/smoke-gpu.py` | Optional GPU smoke (manual) |

## Conclave / jail

This clone lives at `wizards-conclave/.projects/wizards-bard`. That `.projects/` folder is the NTFS jail (`jail.root`). Agents may write anything in `.projects`; they must not write Conclave source, `.env`, or `.conclave`.

Do not recreate `C:/Users/isaac/Documents/wizards-bard`. Conclave skips any `repo:` path outside `.projects/`.

Jail setup (once, elevated, from the Conclave repo):

```powershell
cd C:\Users\isaac\Documents\wizards-conclave
.\scripts\setup-windows-jail.cmd
.\scripts\wz.ps1 doctor
```

Doctor must show PASS for `jail .projects`. Re-run setup after `pnpm install`. `config/projects/bard.yaml` is enabled; goals are implement SPEC.md in phase order.

## Tests

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Default pytest must not load CUDA or ACE-Step weights.
