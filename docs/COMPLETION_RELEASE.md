# Completion release record

The evidence for the release gates in [SPEC.md section 16.8](../SPEC.md). The
bounded release is complete only when every gate below has recorded evidence.
**Status on 2026-10-25: gates 1-3 pass; gate 4 (human listening) is
rendered and awaiting ratings, so completion is not yet established.**

## What changed

Packets 01-11 on branch `completion`, one commit each, listed in
[COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md). Data migrations are
additive and run on start; [COMPLETION_MIGRATION.md](COMPLETION_MIGRATION.md)
covers backup, upgrade, restore, held jobs, and why a downgrade is not
supported.

## Supported setup, as tested

Linux on WSL2 (kernel 6.18.54.1-microsoft-standard-WSL2) with an NVIDIA GeForce
RTX 4070 Ti SUPER 16 GB (driver 610.88), Intel Core i7-13700KF, 19.5 GB RAM
visible to WSL2, Python 3.12.13, torch 2.10.0+cu128, ACE-Step pinned at
`14c0211d5a0653b0f63e27686f4c3f151b4d8629`, and the checkout's checkpoints.
Native Windows and other GPUs were not tested in this release and are
unverified.

## Gate 1: automated (pass)

On the release commit:

| Command | Result |
|---|---|
| `uv run --no-sync python -m pytest -q` | pass (GPU- and network-free; the ACE-Step signature conformance test ran against the installed package) |
| `uv run --no-sync ruff check .`, `ruff format --check .` | pass |
| `uv run --no-sync mypy` | pass |
| `npm run lint --prefix web` | pass |
| `npm test --prefix web` | pass |
| `npm run build --prefix web` | pass |

## Gate 2: real browser (pass, one browser)

Chromium 151.0.7922.34 (headless, Linux, 1600 x 1100), driven by Playwright
against the real server and worker. All nine checks passed:

1. A fresh project starts from an empty brief ("Nothing is locked yet").
2. Prompt to audio, with a page reload while the job was planning: after the
   reload the job strip reattached ("Planning the brief") and the take
   arrived once.
3. Playing a take produced real media playback (currentTime advanced).
4. Space typed into the caption field, and outside text fields it played and
   paused the waveform.
5. Instrumental with written lyrics disabled Generate with the reason; "Keep
   vocals" resolved it, and the take sang the supplied words verbatim.
6. A finished take did not replace the kept take; keeping a take was
   explicit and persisted across a reload.
7. A/B "Switch" paused A and continued B at about the same position; only
   one element played.
8. A job finishing in one project while another was open never appeared in
   the other.
9. The downloaded take was the stored 48 kHz stereo WAV, byte for byte, and
   the export ZIP carried `manifest.json`.

This pass found and fixed a pre-existing bug: at window widths of 1360 px and
up, the takes rail was a closed drawer with no button to open it. Other
browsers and window sizes were not tested.

## Gate 3: live hardware (pass, with one recorded contention failure)

`./scripts/lyre live-check` against real weights. Every default stage passed
across two runs on the same build: preflight, generate, custom, contract,
lyrics, cancel, download, reproducible, profiles (polish and CPU-offloaded
quality), cover, repaint, upload, studio-ops (base-model swaps there and
back), guard, annotations, export, jail, and song. A worker restart was
exercised three times between runs. The opt-in LoRA stage was not run: it is
an hour of training and no shared change touched training.

One cover failed before rendering, right after the XL profile, with
ACE-Step's own pre-flight refusing ("Insufficient free VRAM: need ~0.8 GB,
only 0.5 GB available"). Another process was holding about 5.5 GB of the
card. A fresh worker held the same 9.9 GB, so this was contention, not a
leak; the same stage passed on the next run. The job now fails with advice
that names that cause.

Duration tolerance, agreed before the run: within 1 second of the request
(every rendered take landed exactly on it). The live checker itself uses a
wider 6 s bound.

Checks run live and found two bugs, both fixed in Packet 11:

- A hand-written brief with one planner-filled meta rendered with language
  model codes on, which broke seed reproducibility.
- Instrumental takes ran a lyric-alignment pass on the `[Instrumental]`
  placeholder (about 11 s per take on the quality profile).

Measured timings, swaps, memory, and the tested reproducibility boundary
are in [COMPLETION_BENCHMARK.md](COMPLETION_BENCHMARK.md).

## Gate 4: listening (rendered; awaiting a human)

`scripts/listening_set.py` rendered the six fixed briefs at seeds 11 and 23:
an acoustic lyric song, a rhythmic lyric song (BPM 112 locked), a
supplied-lyrics song, exploration ambience, a combat cue, and a menu theme.
Songs are 60 s and cues 45-60 s, fixed before rendering. All 12 rendered
48 kHz audible audio at exactly the requested length. Locked BPM and preset
tempos were kept, the supplied lyrics were sung verbatim around a
`format_sample` plan, and every cue rendered as an instrumental.

The files, prompts, resolved briefs, and `ratings.md` are in
`output/listening/2026-10-25/` on the test machine (not committed). To close
the gate, a person listens and fills in `ratings.md`: prompt/style fit,
coherence, clipping or glitches, vocals and lyric adherence or the absence
of vocals, and usable starts and endings, each acceptable, minor limitation,
or unusable. The threshold is at least one usable take per brief and no
catastrophic artifact in a kept take. No automated result substitutes for
this.

## Known limitations

- A fixed seed reproduces audio only with language-model codes off. Planned
  briefs and caption rewrite use them, and upstream's sampling is not
  reproducible from the seed.
- On a shared GPU, about 10-11.7 GB of VRAM is needed with the LM resident;
  less headroom makes ACE-Step refuse renders before starting.
- Planning a song takes about 10 s and swapping into the XL profile about
  23 s on this machine; both are upstream behavior.
- Phase 10 (SSE, webhooks, signed artifact tokens) and phase 13 (SPA on
  `/v1`, `/api` deprecation headers) are deferred out of this release.
- A Remote GPU host must be rebuilt from this checkout to apply the new
  creation rules remotely.

## After release

When gate 4 is recorded, mark it in the checklist, cut the release notes from
this record, and stop feature work. Later changes are regressions, security,
installation breakage, and pinned-upstream compatibility (SPEC.md section
16.8).
