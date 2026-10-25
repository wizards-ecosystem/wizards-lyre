# Completion benchmark

Measured for the bounded completion (SPEC.md section 16.8) with
`./scripts/lyre live-check --benchmark`. Each number comes from one take's own
provenance timestamps. Every row is a single sample on one machine. The
numbers are evidence for decisions, not a promised speed: compare runs only on
the same machine with the same inputs.

## Machine and revisions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB (driver 610.88, CUDA UMD 13.3) |
| Shared GPU load | Another process held about 4.9-5.5 GB of VRAM throughout. Lyre had roughly 10.5-11 GB. |
| CPU / RAM | Intel Core i7-13700KF, 24 threads; 19.5 GB visible to WSL2 |
| OS | Linux 6.18.54.1-microsoft-standard-WSL2 (WSL2), ext4, about 140 GB free |
| Python / torch | 3.12.13 / torch 2.10.0+cu128 (CUDA 12.8) |
| ACE-Step | pinned `14c0211d5a0653b0f63e27686f4c3f151b4d8629` |
| Checkpoints | the checkout's `checkpoints/` (turbo, sft, xl-turbo, base, 5Hz LM 1.7B, vae) |
| Lyre | branch `completion` at `e5c84c0` plus the Packet 11 changes |
| Date | 2026-10-25 |

## Startup

| | Seconds |
|---|---|
| Worker process start to ready (iterate DiT + 1.7B LM preloaded), first start | 34 |
| Same, a later restart | up to 69 (measured to the LM's ready line; includes a 1 s poll) |

## Matrix

Two 30-second cases, each with its duration locked: `idea-song` is a planned
song (`from_idea`, the 5 Hz LM planner), and `brief-instrumental` is a
hand-written instrumental brief (`from_brief`, no planning, LM codes off). Rows
run in order, so a `swap` row includes unloading one checkpoint and loading
another. All times are in seconds; "write" is everything from
`generate_music` returning to the take being committed (MP3 encode, hashing,
metadata).

### After the Packet 11 change

| case | profile | swap | load | planning | render + decode | write | submit to commit | peak VRAM reserved (GB) |
|---|---|---|---|---|---|---|---|---|
| idea-song | iterate | no | 0.00 | 10.08 | 6.38 | 2.44 | 19.30 | 11.69 |
| idea-song | iterate | no | 0.00 | 9.70 | 6.23 | 1.58 | 17.70 | 11.70 |
| brief-instrumental | iterate | no | 0.00 | — | 1.47 | 0.38 | 1.94 | 11.62 |
| brief-instrumental | iterate | no | 0.00 | — | 1.52 | 0.37 | 2.25 | 11.62 |
| brief-instrumental | polish | yes | 9.15 | — | 4.01 | 0.33 | 13.63 | 10.82 |
| brief-instrumental | polish | no | 0.00 | — | 3.94 | 0.36 | 4.32 | 10.12 |
| brief-instrumental | iterate | yes | 8.96 | — | 1.53 | 0.32 | 11.04 | 10.81 |
| brief-instrumental | quality | yes | 23.17 | — | 25.49 | 0.47 | 49.56 | 9.48 |
| brief-instrumental | quality | no | 0.00 | — | 10.89 | 0.36 | 11.39 | 9.64 |
| brief-instrumental | iterate | yes | 23.04 | — | 1.49 | 0.33 | 25.14 | 10.28 |

Raw reports for both runs are written to `output/benchmarks/` (not committed).

Every take was 30.0 s of 48 kHz stereo audio. Worker peak RSS was 9.7 GB, and
16.0 GB once the XL profile had been offloaded to CPU.

### Before: the bottleneck

| case | profile | swap | load | planning | render + decode | write | submit to commit | peak VRAM reserved (GB) |
|---|---|---|---|---|---|---|---|---|
| idea-song | iterate | no | 0.00 | 10.10 | 5.79 | 1.07 | 17.06 | 10.15 |
| idea-song | iterate | no | 0.00 | 10.68 | 5.72 | 0.73 | 17.60 | 10.14 |
| brief-instrumental | iterate | no | 0.00 | — | 1.42 | 0.45 | 2.34 | 10.18 |
| brief-instrumental | iterate | no | 0.00 | — | 1.50 | 0.43 | 2.20 | 10.14 |
| brief-instrumental | polish | yes | 8.54 | — | 3.80 | 0.47 | 12.92 | 10.82 |
| brief-instrumental | polish | no | 0.00 | — | 3.86 | 0.50 | 4.66 | 10.15 |
| brief-instrumental | iterate | yes | 6.91 | — | 1.44 | 0.48 | 8.90 | 10.81 |
| brief-instrumental | quality | yes | 24.12 | — | 27.68 | **11.43** | 63.54 | 9.49 |
| brief-instrumental | quality | no | 0.00 | — | 12.35 | **11.22** | 23.81 | 9.65 |
| brief-instrumental | iterate | yes | 16.62 | — | 1.47 | 0.58 | 18.81 | 10.28 |

Writing an instrumental `quality` take took about 11 s. The adapter ran
ACE-Step's lyric-alignment score and LRC timestamping on the `[Instrumental]`
placeholder: a second forward pass with no words to align, which on the
CPU-offloaded XL profile means moving the model back onto the GPU. The change
skips both when there are no sung words. A take with lyrics still gets its
score and LRC. Measured effect: 11.4 s and 11.2 s fell to 0.47 s and 0.36 s
per `quality` instrumental take, and about 0.44 s to 0.37 s on `iterate`.

## What the measurements say

- **Planning dominates a planned song.** The 5 Hz LM's `create_sample`
  takes about 10 s of a 17-19 s `iterate` song. A written brief skips it,
  which is why "another take" of a planned brief costs about 6 s (LM codes on)
  and a hand-written instrumental about 2 s. Planning speed is upstream's;
  Lyre does not change it.
- **Swaps cost 7-9 s between the 2B checkpoints, about 23 s into the XL
  profile, and 17-23 s back out of it** (re-loading the LM and DiT after CPU
  offload). The studio groups nothing to avoid them; they happen only when the
  user changes profile.
- **`quality` renders slower than `iterate`** (11-25 s against 1.5 s for 30 s
  of audio) because of CPU offload. It is labelled as the larger, slower,
  heavier option and is not promised to sound better.
- **Peak VRAM is about 10-11.7 GB with the LM resident.** With another
  process holding 5.5 GB, one 20-second `iterate` cover right after the XL
  profile failed ACE-Step's own VRAM pre-flight ("need ~0.8 GB, only 0.5 GB
  available"). A fresh worker held the same 9.9 GB, so this was contention,
  not a leak. A shared GPU needs that headroom.
- **Write, commit, and fetch are small.** The staged commit, hashing, and
  manifest work add well under a second; the client's audio fetch after `done`
  took 10-50 ms on loopback.
- **Studio polling** is one job GET per second per active job (every 5 s
  after ten minutes), a project job list every 5 s while idle, and health plus
  capabilities every 5 s. Measured server work per poll was negligible next
  to rendering; no change was made.

## Reproducibility (tested boundary)

Seeds were checked on the same machine and checkpoints:

- A hand-written brief with LM codes off (`caption_rewrite` off, every meta
  set) re-renders byte-identically from the same seed (`live-check` stage
  `reproducible`).
- A planned brief has LM codes on. Two "another take" renders of the same
  brief at the same seed produced different audio, and re-planning the same
  idea gave a different caption. ACE-Step's LM sampling is not reproducible
  from the seed. "Replay settings" says so for such takes.

## Remaining

Planning time, swap time, and the offloaded XL path are upstream behavior and
are documented rather than changed. A run on a GPU with no other tenant would
show more VRAM headroom than this one had.
