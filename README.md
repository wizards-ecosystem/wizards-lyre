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

This repo is a Wizard's Conclave project (`config/projects/bard.yaml` in wizards-conclave). Keep that YAML `enabled: false` until the Windows jail canary passes:

```powershell
# Elevated PowerShell
cd C:\Users\isaac\Documents\wizards-conclave
powershell -File scripts\install-windows-jail.ps1 -Root C:\Users\isaac\Documents\wizards-bard
.\scripts\wz.ps1 doctor
```

When doctor reports the bard jail user can write this folder and cannot write the Conclave repo, set `enabled: true`.

## Tests

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Default pytest must not load CUDA or ACE-Step weights.
