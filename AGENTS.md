# Agent rules: The Wizard's Lyre

Rules for coding agents working in this repository. Human contributors want
[CONTRIBUTING.md](CONTRIBUTING.md), which covers the same scope constraints
alongside setup and the test loop.

- **SPEC.md is the sole product spec.** Implement it in the phase order written
  there. Do not re-open product-scope research.
- Generation is **ACE-Step 1.5 on the local GPU only**. No Lyria, Gemini music,
  ElevenLabs Music, Stability Audio, Magenta RealTime, LeVo, YuE, or unofficial
  Suno/Udio clients. Do not add stubs or optional adapters for those.
  `tests/test_spec_lock.py` enforces this by scanning the source.
- Do not ship ACE-Step's Gradio UI. Call the ACE-Step Python API from
  `worker/acestep_worker/`.
- Bind **127.0.0.1** only. No auth. No cloud deploy. No Docker.
- Default tests mock the worker. **`pytest` must never require a GPU** and must
  never import `acestep` or `torch` at module scope.
- If upstream ACE-Step parameter names change, adapt inside the worker adapter
  and keep Lyre's HTTP schema stable.
- `server/storage`, `server/jobs`, and `worker/acestep_worker` are packages
  whose `__init__` re-exports their modules' surface. When patching one of
  these in a test, patch the **defining module**, not the package re-export;
  patching the re-export leaves internal callers on the real implementation and
  the test will pass while exercising nothing.
