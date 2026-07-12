# 8da4w E2E-Ranked Tile/Subgroup Sweep: Final Report

**Feature**: `specs/027-e2e-tile-sweep` | **Date**: 2026-07-11 | **Target**: M5 EVT1
(`xgpusw-debug08`, driver `f14c51b6f8`/`c9861e9906d03fa2c7d48b804e1a1c80`, clocks pinned
509/2730/663 MHz), Llama 3.1 8B `8da4w` buffer PTE, 2048-token prefill.

> **SHIPPED 2026-07-12** (commit `42aabb4e0` on `yanwen/dev-1.3`): this feature's winner
> is now the production default `8da4w` tile/loop configuration on `dev`. Full-stack
> (SDPA coopmat default-on) validation confirmed **+12.5%** (131.24 → 147.65 tok/s),
> larger than the +9.32% measured on the SDPA-less validation branch below — see
> `results/dev-branch-production-validation.md`. The rest of this report describes the
> original sweep that found the winner; the Recommendation section's Tier-2 ask is now
> satisfied.

## FinalAnswer

**The current e2e winner is `tsweep_t64x32k32g12s64`** —
`WG_TILE_M=64, WG_TILE_N=32, WG_TILE_K=32, SG_GRID_X=1, SG_GRID_Y=2, SUBGROUP_SIZE=64`,
`dbuf2` loop. Confirmed 3-run mean **110.02 tok/s** prefill vs. the shipped baseline's
**100.65 tok/s** — **+9.32%**, CoV 0.48%, non-overlapping 3-run ranges (baseline
100.28–100.92, winner 109.47–110.51). This is `specs/025`'s **rank-3** microbenchmark
candidate (1570.05 GFLOP/s, well behind `025`'s own #1 and `026`'s #1) — neither prior
feature's microbenchmark-ranked "winner" is the actual e2e winner.

A close second, also confirmed beating baseline: `tsweep_t64x32k32g21s64` (`SG_GRID_X=2,
SG_GRID_Y=1`, otherwise identical tile) at **108.53 tok/s**, **+7.84%**, CoV 0.11%.

## Microbenchmark-vs-e2e rank agreement: **DISAGREE**

| E2E rank | Token | Mean tok/s | vs baseline | Microbenchmark rank (of 27) |
|---|---|---|---|---|
| 1 | **`t64x32k32g12s64`** | **110.02** | **+9.32%** | 4 |
| 2 | `t64x32k32g21s64` | 108.53 | +7.84% | 3 |
| — | **BASELINE (shipped)** | **100.65** | — | not ranked (not a swept candidate) |
| 3 | `t64x64k16g21s32` (`026`'s #1, 2207.2 GFLOP/s) | 98.10 | -2.53% | **1** |
| 4 | `t64x32k16g21s64` | 95.83 | -4.79% | 6 |
| 5 | `t64x32k16g12s64` | 93.60 | -7.00% | 8 |
| 6 | `t64x16k32g12s64` | 92.07 | -8.52% | 5 |
| 7 | `t128x32k16g12s64` (`025`'s #1, 1736.05 GFLOP/s) | 91.15 | **-9.43%** | **2** |
| (skipped) | `t64x64k16g12s64` | (screened -14.4%, not escalated) | — | 7 |

The microbenchmark's top-2 candidates (`026`'s subgroup=32 winner at rank 1, `025`'s
subgroup=64 winner at rank 2) are e2e ranks **3 and 7 of 8** — `025`'s own actual
microbenchmark champion is the single **worst**-performing candidate end-to-end of
everything measured. Conversely, the two actual e2e winners rank only 3rd and 4th on the
microbenchmark. This is not a subtle disagreement — it's close to inverted at the top,
confirming (more sharply than `026`'s single-point finding already suggested) that
isolated-kernel GFLOP/s is a poor predictor of this workload's real end-to-end throughput
on this hardware.

## Screening stage (1 run each, 8 candidates + baseline)

| Token | Screen tok/s | Screen ratio vs baseline | Escalated? |
|---|---|---|---|
| BASELINE | 100.73 | — | (always confirmed) |
| `t64x32k32g12s64` | 111.57 | +10.8% | yes |
| `t64x32k32g21s64` | 107.76 | +7.0% | yes |
| `t64x64k16g21s32` | 98.29 | -2.4% | yes |
| `t64x32k16g21s64` | 94.87 | -5.8% | yes |
| `t64x32k16g12s64` | 92.95 | -7.7% | yes |
| `t64x16k32g12s64` | 92.84 | -7.8% | yes |
| `t128x32k16g12s64` | 91.90 | -8.8% | yes |
| `t64x64k16g12s64` | 86.24 | **-14.4%** | **no** (below -10% threshold — device time saved) |

7 of 8 candidates escalated (only 1 was clearly far enough behind to skip) — the adaptive
bar still saved 3 runs on that one candidate, and correctly did not filter out either
eventual winner.

## Confirmation stage (3 fresh runs each)

See table in FinalAnswer/rank-agreement sections above for the ranked view; full per-run
data in `results/confirmation_results.json`.

## Search cost (SC-004)

- **9 candidates measured end-to-end** (8 shortlisted + baseline): 9 screening runs + 7×3
  confirmation runs (baseline + 6 escalated, since 1 of 8 shortlisted candidates was not
  escalated) = 30 total e2e runs.
- Zero candidates required building new shader variants — all 8 shortlisted candidates
  already existed as built binaries from `specs/025`/`specs/026`.
- **User Story 2 was NOT triggered** (spec FR-006) — a real, confirmed winner was found in
  User Story 1, so no search extension was needed. `T016`'s gate check applies: skip.

## Skip-reasons appendix (SC-005)

19 correctness-passing candidates from the combined `025`+`026` pool (27 total) were NOT
shortlisted, all for the same documented reason: ranked below the top-8 cutoff by
microbenchmark score (see `results/prefilter_ranking.json` for the full 27-candidate
ranking with each one's `microbenchmark_rank`). None were excluded for correctness
failures within this feature's own filtering — correctness filtering already happened
upstream in `025`/`026`.

## Scope note (methodology correction during this feature)

`plan.md`'s and `research.md`'s original Decision 2 assumed `025`'s `round2_results.json`
`avg_gflops` was measured purely on 8B-shaped GEMMs. On inspection during implementation,
it is actually a **FLOP-weighted average across all three model sizes' `wq`+`w1_gate`
shapes (1B/3B/8B)**, not an 8B-only number. This feature's e2e validation still used the
8B model exclusively (the largest-FLOP, already-established validation target from this
session's `026` Tier-2 check) — a reasonable single representative point, but not a strict
per-shape match to the averaged microbenchmark score. This is disclosed here rather than
silently treated as resolved; a more rigorous follow-up could validate the same 8
candidates on 1B and 3B as well to check whether the ranking (and in particular the
winner) holds across all three model sizes, not just 8B.

## Recommendation

`tsweep_t64x32k32g12s64` is a real, statistically confirmed (+9.32%, non-overlapping 3-run
ranges) end-to-end improvement over the shipped baseline — the first configuration in this
workstream's `025`→`026`→`027` sequence to actually win on the metric that matters.

**Update 2026-07-12: shipped.** Applied to `dev`'s production shader (commit `42aabb4e0`)
and re-validated on the full stack (SDPA coopmat included) — see `results/
dev-branch-production-validation.md`. Outstanding from the scope note above: validating
this winner (and its close second) on the 1B/3B models is still a good follow-up, but is
no longer a blocker for shipping the 8B result, since the 8B model is this workstream's
primary target shape family.
