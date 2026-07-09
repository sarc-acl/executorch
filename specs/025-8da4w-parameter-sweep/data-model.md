# Phase 1 Data Model: 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep

This feature is file-based (JSON/CSV/Markdown artifacts under
`specs/025-8da4w-parameter-sweep/results/`), not a database-backed system. The entities
below describe the shape of those files.

## LoopStructureResult

One User-Story-1 outcome for a `dbuf{1..4}` loop-structure variant, measured at the
currently-shipped 128×64/K32/2×2/s64 tile/subgroup geometry, before any geometry sweep.

| Field | Type | Notes |
|---|---|---|
| `variant` | enum | `dbuf1` \| `dbuf2` \| `dbuf3` \| `dbuf4`. |
| `dispatch_confirmed` | bool | True only if kernel-name capture confirms the coopmat kernel dispatched (Principle VI), not a tiled fallback. |
| `correctness_status` | enum | `pass` \| `fail` \| `not_attempted`. |
| `mean_us` | float \| null | 3-run mean, only populated if `correctness_status == pass`. |
| `cov` | float \| null | Coefficient of variation across the 3 runs (spec Clarified 2026-07-09: must be < 5%). |
| `driver_hash` | string | Verified driver identity at measurement time (Principle VIII). |
| `clocks_pinned` | bool | Whether the pin was verified bound (Principle VII). |
| `failure_reason` | string \| null | Populated if not measured to completion, e.g. `"pipeline_creation_crash"`. |

**Derived**: `fastest_variant` = the `LoopStructureResult` with the lowest `mean_us` among
those with `correctness_status == pass`. `matches_user_claim` = `fastest_variant.variant == "dbuf2"`
(spec SC-001; must be reported explicitly either way).

## ConfigurationCandidate

One point in the `8da4w` tile/subgroup search space, at the fixed loop structure from
`LoopStructureResult.fastest_variant`. `SUBGROUP_SIZE` is fixed at 64 for every candidate
(research.md Decision 1 — 32 crashes the Xclipse PAL compiler for int8 WMMA); this is a
narrower space than `022`'s `4w` enumeration, not a copy of it.

| Field | Type | Notes |
|---|---|---|
| `wg_tile_m` | int | Output tile height. |
| `wg_tile_n` | int | Output tile width. |
| `wg_tile_k` | int | K-step per loop iteration; must divide the INT4 group size. |
| `sg_grid_x` | int | Subgroups tiling N. |
| `sg_grid_y` | int | Subgroups tiling M. |
| `subgroup_size` | int | Fixed at 64 for every candidate (research.md Decision 1). |
| `token` | string | Derived `ET_VK_DQ8CA_COOPMAT_TILE_VARIANT` value, e.g. `tsweep_t128x64k32g22s64`. |
| `wg_size` | int (derived) | `sg_grid_x * sg_grid_y * subgroup_size`. |
| `lds_bytes` | int (derived) | Double-buffered `Ash_int8`/`Bsh_int8` footprint **plus** the `izp_sh`/`ifs_sh`/`wsum_sh`/`wsc_sh`/`bias_sh` broadcast arrays this shader carries that `4w`'s does not (research.md Decision 1/2). |
| `accumulators_per_sg` | int (derived) | `(wg_tile_m/sg_grid_y/16) * (wg_tile_n/sg_grid_x/16)`; this shader carries this count in **both** an `int32` and an `fp32` accumulator array simultaneously. |
| `valid` | bool (derived) | Passes all constraints in Validation rules below. |
| `compile_status` | enum | `not_attempted` \| `compiles` \| `compile_failed`. Only known once actually built. |

**Validation rules**:
- `wg_size <= 1024`
- `wg_tile_m % (sg_grid_y * 16) == 0` and `wg_tile_n % (sg_grid_x * 16) == 0`
- `wg_tile_k` divides the INT4 group size
- `lds_bytes <= 65536`
- `subgroup_size == 64` (any candidate generated with `subgroup_size == 32` is rejected at
  enumeration time as a known driver-crash configuration, not measured — research.md
  Decision 1)

## AnalyticalScore

A pre-measurement ranking value attached to a `ConfigurationCandidate`. Never itself
reported as a performance result (spec Key Entities).

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `occupancy_proxy` | float | `min(65536/lds_bytes, 1024/wg_size)`, using this shader's own `lds_bytes` (research.md Decision 2 — not `4w`'s formula inputs). |
| `register_penalty` | float | `1 + max(0, accumulators_per_sg - K) * weight`, with `K`/`weight` recalibrated from `LoopStructureResult`'s shipped-geometry measurement (research.md Decision 2), not `022`'s `4w`-calibrated `K=8`/`0.15`. |
| `score` | float | `occupancy_proxy / register_penalty`. |
| `rank` | int | 1-indexed rank among all legal `8da4w` candidates by `score` desc. |
| `shortlisted` | bool | True if in the top-ranked shortlist, or force-included as an anchor. |
| `shortlist_reason` | string | `"top-rank"` \| `"anchor:shipped-config"` \| `"anchor:4w-winner"` \| `"known_compile_failure"` \| `"excluded"`. |

## MeasurementResult

One on-device outcome for a `ConfigurationCandidate`, at a specific search round.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `round` | enum | `round1_gate` \| `round2_full_shapes` \| `round3_confirmation`. |
| `correctness_status` | enum | `pass` \| `fail` \| `skipped_oversized_shape`. |
| `shapes_measured` | list[(K,N)] | Which of the 6 representative shapes (`wq`+`w1_gate` × {1B,3B,8B}) this round covered. |
| `gflops_per_shape` | map[(K,N) -> float] | FLOP-weighted throughput per shape. |
| `mean_gflops` | float | Only populated for `round3_confirmation`. |
| `stddev_gflops` | float | Only populated for `round3_confirmation` — must correspond to a 3-run mean with CoV<5% (spec Clarified 2026-07-09). |
| `run_count` | int | Number of independent process invocations behind `mean`/`stddev`; must be 3 for `round3_confirmation`. |
| `driver_hash` | string | Verified driver identity at this round (Principle VIII). |
| `clocks_pinned` | bool | Whether the pin was verified bound for this round (Principle VII). |
| `eliminated_at` | bool | True if this result caused the candidate to be dropped from the next round. |
| `elimination_reason` | string \| null | e.g. `"compile_failed"`, `"correctness_failed"`, `"below round-2 top-third cutoff"`. |

## SearchBudget

Tracks consumption against spec SC-002/FR-007's caps (Clarified 2026-07-09: ≤15% of the
legal `8da4w` space, hard-capped at 30 on-device measurements).

| Field | Type | Notes |
|---|---|---|
| `total_valid_universe` | int | Size of the re-derived legal `8da4w` space (research.md Decision 1) — a Phase 0/1 output, not assumed equal to `4w`'s 642. |
| `budget_cap` | int (derived) | `min(round(0.15 * total_valid_universe), 30)`. |
| `configs_measured_on_hardware` | int | Distinct candidates with ≥1 `MeasurementResult`, excluding `LoopStructureResult`s and the Decision 1 subgroup=32 re-verification anchor (neither counts against this budget). Must stay `<= budget_cap`. |
| `total_device_seconds` | float | Sum of on-device wall-clock time across all rounds. |
| `estimated_exhaustive_device_seconds` | float | Estimated cost of measuring all `total_valid_universe` candidates at Round-2 rigor, for the SC-006 5x-reduction comparison. |
| `budget_exceeded` | bool | True if `configs_measured_on_hardware > budget_cap` at any point — a hard stop condition. |

## OptimalConfiguration

The final recommended candidate (or an explicit "no improvement found" outcome per spec
FR-010).

| Field | Type | Notes |
|---|---|---|
| `loop_structure` | LoopStructureResult | The User Story 1 winner this candidate was measured under. |
| `candidate_token` | string \| null | Null if no candidate beat the shipped baseline (FR-010 case). |
| `round3_result` | MeasurementResult \| null | The confirming measurement (3-run mean, CoV<5%). |
| `comparison_vs_shipped_8da4w` | float \| null | Speedup ratio, FLOP-weighted, vs the currently-shipped `8da4w` configuration. |
| `comparison_vs_4w_winner` | float \| null | Speedup ratio vs `4w`'s 128×64/K16/2×2/s32 winner, FLOP-weighted (spec FR-006). |
| `tie_broken` | bool | Whether a documented tie-breaking rule was invoked (spec Acceptance Scenario, User Story 3). |
| `spirv_verified` | bool | Whether the compiled SPIR-V was inspected for genuine int8 cooperative-matrix instructions (Principle VI). |
| `recommendation` | enum | `"productionize_candidate"` \| `"keep_existing_winner"`. |
