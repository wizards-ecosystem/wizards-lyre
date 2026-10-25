# Completion checklist

The bounded completion in [SPEC.md section 16](../SPEC.md#16-bounded-completion-amended-2026-10-25),
as one list. Each packet is a separate reviewable change, in this order. A
packet is checked only when its acceptance criteria have evidence in the
repository (tests, or a recorded report for the live gates).

| # | Packet | Priority | Depends on | Status |
|---|---|---|---|---|
| 01 | Adopt the bounded contract (SPEC.md section 16, AGENTS.md) | P0 | none | done |
| 02 | Resolve ideas and explicit fields consistently | P0 | 01 | done |
| 03 | Freeze submissions and complete provenance | P0 | 02 | done |
| 04 | Make recovery and edits safe together | P0 | 03 | done |
| 05 | Restore and observe every core job (browser) | P0 | 03, 04 | done |
| 06 | Make the creation screen coherent | P1 | 02, 05 | done |
| 07 | Truthful readiness and curated presets | P1 | 02, 06 | done |
| 08 | Finish audition, keep, and reuse | P1 | 03, 06 | done (real-browser media check is gate 2) |
| 09 | Export manifest and narrow agent parity | P1 | 03, 05, 08 | done |
| 10 | Preserve existing lanes and migrate safely | P0 gate | 04–09 | open |
| 11 | Measure, optimize, simplify touched code | P1 | 10 | open |
| 12 | Verify, freeze, release | P1 gate | 01–11 | open |

## Release gates (SPEC.md section 16.8)

- [ ] Automated: lint, type, test, and build commands pass on the release commit.
- [ ] Real browser pass, with browser versions recorded.
- [ ] Live hardware gate on the 16 GB Linux/WSL2 baseline with real weights.
- [ ] Human listening gate: six briefs, two seeds each, ratings recorded.

A gate without recorded evidence is open. P2 items (standalone brief preview,
serial alternate takes) never block a release whose P0 and P1 gates pass.
