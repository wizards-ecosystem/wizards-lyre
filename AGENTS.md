# Agent rules — Wizard's Bard

- **SPEC.md is the sole product spec.** Implement it in the phase order written there.
- Generation is **ACE-Step 1.5 on the local GPU only**. No Lyria, Gemini music, ElevenLabs Music, Stability Audio, Magenta RealTime, LeVo, YuE, or unofficial Suno/Udio clients. Do not add stubs for those.
- Do not ship ACE-Step's Gradio UI. Call `acestep.inference.generate_music` from `worker/`.
- Bind **127.0.0.1** only. No auth. No cloud deploy.
- Default tests mock the worker. Do not require a GPU for `pytest`.
- Do not re-open product-scope research. If upstream ACE-Step parameter names change, adapt in the worker and keep Bard's HTTP schema stable.
- Canonical clone is `/home/limb06/wizards-bard` (WSL Ubuntu). Conclave's Windows `.projects/` jail does not apply here.
