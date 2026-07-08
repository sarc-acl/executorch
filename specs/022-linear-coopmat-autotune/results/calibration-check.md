# Calibration Check (T009): Analytical Score vs. Known Real Measurements

Scored the 10 configurations in `known-measurements.json` with the
Decision-2 formula (`score = occupancy_proxy / register_penalty`) and
compared against their real, on-device speedup vs. dbuf1.

| token | analytical score | rank (of 9 compiling) | real speedup vs dbuf1 | real rank |
|---|---|---|---|---|
| tsweep_t64x64k16g22s32 | 6.10 | 1 | 1.15x | 3 |
| tsweep_t64x128k16g22s32 | 4.41 | 2 | 1.18x | 2 |
| **tsweep_t128x64k16g22s32 (winner)** | 3.88 | 3 | **1.25x** | **1** |
| tsweep_t64x64k32g22s64 | 3.37 | 4 | 0.91x | 7 |
| tsweep_t128x128k16g42s32 (dbuf1) | 3.12 | 5 | 1.00x | 5 |
| tsweep_t128x128k16g22s64 | 1.42 | 6 | 0.95x | 6 |
| tsweep_t128x64k64g44s32 | 1.19 | 7 | 0.88x | 8 |
| tsweep_t128x256k16g42s32 | 1.02 | 8 | 1.14x | 4 |
| tsweep_t256x256k16g44s32 | 0.72 | 9 | 0.47x | 9 |
| tsweep_t128x64k16g44s32 | N/A (compile_failed) | — | N/A | — |

## Outcome: PARTIAL PASS — model revised, not the raw formula

**What the model gets right**: the two known worst performers
(256×256/4×4, rank 9; and 128×64/K64/4×4, rank 7) both land in the bottom
half analytically too (ranks 9 and 7 of 9) — the occupancy/LDS-based
proxy correctly flags oversized/LDS-heavy tiles as risky, which is this
model's main job (eliminating clearly-bad candidates before they consume
device time).

**What the model gets wrong**: it does not reliably identify the single
best performer. The true winner (128×64/K16/2×2, real rank 1) scores only
3rd, behind two configs that are real but smaller wins (64×64/K16/2×2 and
64×128/K16/2×2). More importantly, **128×256/K16/4×2 — a real, solid
mid-pack performer (1.14x, real rank 4 of 9)** — scores 8th of 9
analytically, because its 16-accumulator/28.5KB-LDS profile trips the
occupancy/register penalty harder than its real throughput justifies. A
naive top-28-by-score shortlist would very plausibly have dropped this
real, decent-performing config.

**Root cause**: the occupancy proxy rewards small tiles for higher per-CU
residency but has no term for "a tile too small does less useful work per
dispatch" — a genuine GEMM tuning tradeoff (occupancy vs. work-per-launch)
this simple, unfitted heuristic doesn't model. Retuning constants
(register-penalty threshold/slope, occupancy weighting) to fix this
specific case was rejected — with only 10 points, any fix would be fit to
noise, not signal (research.md Decision 2's own stated risk).

## Resolution (research.md Decision 3, revised)

Instead of touching the formula, broadened the force-include rule: **all
9 previously-measured, compiling known configurations** are shortlisted
regardless of analytical score (not just the 2 originally-planned
anchors). The 1 known compile failure is excluded with an explicit
`known_compile_failure` reason rather than silently dropped or
re-attempted. This directly fixes the demonstrated failure mode (real data
overridden by an imperfect heuristic) without any risk of overfitting the
scoring formula itself — the formula's actual job, eliminating the
clear-worst candidates from the *unmeasured* remainder of the 642-config
universe, is unaffected and still directionally validated by the two
correctly-identified worst performers above.
