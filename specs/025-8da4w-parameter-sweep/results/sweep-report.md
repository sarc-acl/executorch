# M5 EVT1 `8da4w` (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep

**Feature**: `specs/025-8da4w-parameter-sweep` | **Date**: 2026-07-09 | **Target**: M5 EVT1
(Exynos 2500 / Xclipse 970), driver `f14c51b6f8` (md5 `c9861e9906d0…`), clocks pinned
509/2730/663 MHz (verified bound before every measurement round)

## Result at a glance

**Optimal `8da4w` configuration found: `128×64→32/K16/1×2/s64`** (`WG_TILE_M=128,
WG_TILE_N=32, WG_TILE_K=16, SG_GRID_X=1, SG_GRID_Y=2, SUBGROUP_SIZE=64`, loop structure
`dbuf2`), measured at **1731.0 GFLOP/s** (3-run mean, CoV 0.14%) — **+2.55% faster** than
the currently-shipped `128×64/K32/2×2/s64` configuration (1688.1 GFLOP/s).

## SC-001: Loop-structure re-confirmation

Fresh, independent re-measurement on 2026-07-09 (not a reuse of `specs/023`'s 2026-07-08
numbers) **confirms the user-reported `dbuf2` claim**: `dbuf2` is fastest on all 6/6
representative shapes.

| Variant | Avg GFLOP/s (6 shapes) | vs. shipped dbuf4 |
|---|---|---|
| dbuf1 | 1280.2 | -11.6% |
| **dbuf2** | **1688.1** | **+16.5%** |
| dbuf3 | 1539.6 | +6.3% |
| dbuf4 (shipped) | 1448.7 | — |

Matches `specs/023`'s own finding (dbuf2 wins 6/6, ~+18% vs dbuf4) within normal run-to-run
device variance. `dbuf2` is held fixed for the tile/subgroup sweep below.

## SC-002/FR-007: Search budget

- Legal `8da4w` tile/subgroup space (subgroup_size fixed at 64, per research.md Decision 1;
  `wg_tile_k=64` variants further excluded once the B-staging pass-count constraint was
  discovered — see "Findings" below): **542 candidates**.
- Budget cap: `min(round(0.15×542), 30)` = **30** (29 top-rank/anchor candidates measured on
  hardware + the 1 shipped-config anchor reused from `dbuf_reconfirmation.json`, not
  re-measured).
- Actual candidates taken to real on-device measurement: **29** — **5.4%** of the legal
  space, well under the 15%/30 cap.
- 4 of 29 were eliminated at Round 1 (correctness gate) for having `WG_TILE_K=64`, which is
  never divisible by this shader's real production `group_size=32` — see "Findings."
- 25 candidates received the full 6-shape Round 2 measurement; the top 5 received Round 3
  (3-run mean/CoV) confirmation.

## SC-002/SC-004: Fastest configuration, full ranking

Round 2 (25 candidates that passed correctness), full 6-shape mean GFLOP/s:

| Rank | Token | Tile (M×N/K, grid) | Avg GFLOP/s |
|---|---|---|---|
| 1 | `tsweep_t128x32k16g12s64` | 128×32/K16, 1×2 | **1736.0** |
| 2 | `tsweep_t64x32k32g21s64` | 64×32/K32, 2×1 | 1570.0 |
| 3 | `tsweep_t64x32k32g12s64` | 64×32/K32, 1×2 | 1551.4 |
| 4 | `tsweep_t64x16k32g12s64` | 64×16/K32, 1×2 | 1491.4 |
| 5 | `tsweep_t64x32k16g21s64` | 64×32/K16, 2×1 | 1475.5 |
| 6-25 | (see `round2_results.json`) | | 805.2 - 1472.2 |

Round 3 (3-run mean, CoV<5% per spec Clarified 2026-07-09) confirmation of the top 5:

| Token | Round 3 mean GFLOP/s | CoV | Run 1 / 2 / 3 |
|---|---|---|---|
| **`tsweep_t128x32k16g12s64`** | **1731.0** | **0.14%** | 1728.4 / 1731.4 / 1733.2 |
| `tsweep_t64x32k32g21s64` | 1570.0 | 0.28% | 1571.3 / 1565.1 / 1573.7 |
| `tsweep_t64x32k32g12s64` | 1552.1 | 0.17% | 1552.5 / 1549.3 / 1554.4 |
| `tsweep_t64x16k32g12s64` | 1493.7 | 0.07% | 1494.4 / 1494.1 / 1492.5 |
| `tsweep_t64x32k16g21s64` | 1472.9 | 0.29% | 1476.9 / 1468.4 / 1473.6 |

The ranking is stable between Round 2 and Round 3 (same order, <1% shift) — a confident
result, not noise. `tsweep_t128x32k16g12s64` wins by a clear margin over the runner-up
(+10.3%), well outside the ~0.2-0.3% run-to-run CoV of either.

## FR-006/SC-004: Comparison against baselines

| Configuration | GFLOP/s | vs. shipped 8da4w |
|---|---|---|
| Shipped `8da4w` (`128×64/K32/2×2/s64`, dbuf2 loop) | 1688.1 | — |
| **Winner: `128×32/K16/1×2/s64`, dbuf2 loop** | **1731.0** | **+2.55%** |
| `4w`'s actual autotuned best (`specs/022`, `128×64/K16/1×4/s32`) | 2518.77 | n/a (different shader/precision) |

`4w`'s fp16 coopmat shader remains substantially faster in absolute terms (2518.77 vs
1731.0 GFLOP/s, ~1.45x) even after this sweep's improvement — consistent with
`specs/024-8da4w-slower-than-4w`'s premise that `8da4w` underperforms `4w` on this hardware.
This sweep found a real, if modest, `8da4w`-internal improvement; it does not close that
gap. (Note: the specific "128×64/K16/2×2/s32" anchor the user's original request named is
`022`'s prior, sub-optimal 7-config-sweep result, superseded within `022` itself by the
1×4-grid config shown above — both are `4w`-only reference points, included per this
feature's Clarifications for cross-shader context, not because either is expected to be
legal or optimal for `8da4w`'s different int8-MMA register/shared-memory footprint.)

## SC-003: Correctness and dispatch verification

- Every reported GFLOP/s number is gated on a passing small-shape (`M=K=N=128`) correctness
  check against the CPU/tiled reference — zero failed configurations appear in the ranking
  above.
- The winner's dispatch was confirmed via kernel-name capture
  (`linear_dq8ca_q4gsw_coopmat_tsweep_t128x32k16g12s64_buffer_texture2d_half`) — genuine
  coopmat dispatch, not a tiled fallback.
- SPIR-V inspection (`spirv-dis`) of the winner confirms 8
  `OpCooperativeMatrixMulAddKHR` sites and an int8 (`%char`) `OpTypeCooperativeMatrixKHR`
  component type — genuine int8 WMMA, not a mislabeled fallback (Constitution Principle VI).

## Findings (beyond this feature's core question)

1. **The `8da4w` legal tile/subgroup space is genuinely smaller than assumed twice over**:
   first, `SUBGROUP_SIZE=32` is illegal (research.md Decision 1, pre-existing knowledge);
   second, **`WG_TILE_K=64` is illegal at this workstream's real production `group_size=32`**
   (a constraint this feature's own `tile_constraints.py` initially missed — its first draft
   used a default `group_size=128`, which incorrectly allowed `WG_TILE_K=64`). Caught by
   real compile/correctness failures during Round 1 (`tsweep_t{16x32,32x16,32x32}k64...`),
   not by inspection — exactly the failure mode spec Edge Cases anticipated ("candidate
   mathematically incompatible with quantization group size ... caught by the correctness
   check"). `tile_constraints.py`'s default was corrected to `group_size=32` after this was
   found; the legal universe shrank from 609 to 542 candidates as a result.
2. **The documented `SUBGROUP_SIZE=32` Xclipse PAL compiler crash did not reproduce** on the
   current driver (`c9861e9906…`) for the shipped tile shape — see
   `results/subgroup32-reverification.md`. Flagged as a finding for a follow-up feature, not
   acted on in this sweep's own search space (research.md Decision 1).
3. A pre-existing, unrelated correctness bug was independently discovered in the
   `linear_dq8ca_q4gsw_tiled` fallback path at the `M=K=N=128` shape (reproduces even with
   `ET_VK_FORCE_TILED_LINEAR=1` and no sweep variant active) — this is what actually caused
   the 4 `WG_TILE_K=64` eliminations to report FAILED rather than a clean ineligibility skip.
   Out of scope for this feature to fix; worth a follow-up bug report.

## Recommendation

`recommendation: productionize_candidate` — `128×32/K16/1×2/s64` (dbuf2 loop) is a real,
statistically confident (+2.55%, CoV<0.3%) improvement over the shipped `8da4w`
configuration, validated for correctness and genuine coopmat dispatch. This is Tier-1
(shader microbenchmark) evidence only, per this feature's scope — a Tier-2 (`.pte`
end-to-end tok/s) validation is recommended before this configuration is shipped by default,
per Constitution Principle IV.

## Search cost (SC-006)

- 29 real on-device measurement rounds (Round 1 correctness gate) + 25 Round 2 full-shape
  passes + 15 Round 3 confirmation runs (5 candidates × 3 runs) = well under an exhaustive
  542-candidate × 6-shape × 3-run campaign, which would require on the order of 40-60x more
  on-device time at this same per-candidate cost.
- Total distinct candidates measured on hardware: **29 / 542 = 5.4%** of the legal universe.

## Pruning audit

Full ranking, inclusion/exclusion reasons, and compile/correctness status for every one of
the 542 legal candidates are in `configs.json` (full enumeration) and `shortlist.json`
(analytical ranking + shortlist reason for every candidate, per spec FR-009/SC-005).
