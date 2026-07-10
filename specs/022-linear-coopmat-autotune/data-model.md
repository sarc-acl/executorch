# Phase 1 Data Model: Smart Autotuning for q4gsw CoopMat Tile Configuration

This feature is file-based (JSON/CSV/Markdown artifacts under
`specs/022-linear-coopmat-autotune/results/`), not a database-backed system.
The entities below describe the shape of those files.

## ConfigurationCandidate

One point in the tile-geometry search space. Always buffer weight storage
(per the standing scope decision), always the dbuf1 loop-structure shape.

| Field | Type | Notes |
|---|---|---|
| `wg_tile_m` | int | Output tile height. One of {16,32,64,128,256}. |
| `wg_tile_n` | int | Output tile width. One of {16,32,64,128,256}. |
| `wg_tile_k` | int | K-step per loop iteration. One of {8,16,32,64,128}; must divide group_size=128. |
| `sg_grid_x` | int | Subgroups tiling N. One of {1,2,4,8}. |
| `sg_grid_y` | int | Subgroups tiling M. One of {1,2,4,8}. |
| `subgroup_size` | int | 32 or 64 (HW-supported range on Xclipse 970). |
| `token` | string | Derived `ET_VK_Q4GSW_COOPMAT_VARIANT` value, e.g. `tsweep_t128x64k16g22s32`. |
| `wg_size` | int (derived) | `sg_grid_x * sg_grid_y * subgroup_size`; threads per workgroup. |
| `lds_bytes` | int (derived) | Double-buffered `Ash`+`Bsh` shared-memory footprint. |
| `accumulators_per_sg` | int (derived) | `(wg_tile_m/sg_grid_y/16) * (wg_tile_n/sg_grid_x/16)`. |
| `valid` | bool (derived) | Passes all four known constraints (§ spec Assumptions / this session's constraint model). |
| `compile_status` | enum | `not_attempted` \| `compiles` \| `compile_failed`. Only known once actually built. |

**Validation rules** (mirrors the constraint model already validated against
10 real on-device results this session):
- `wg_size <= 1024`
- `wg_tile_m % (sg_grid_y * 16) == 0` and `wg_tile_n % (sg_grid_x * 16) == 0`
- Staging pass counts (`A_PASSES`, `B_PASSES`, derived from `wg_size` and
  tile dims) must both be positive integers
- `lds_bytes <= 65536`

## AnalyticalScore

A pre-measurement ranking value attached to a `ConfigurationCandidate`.
Never itself reported as a performance result (spec Key Entities).

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `occupancy_proxy` | float | `min(65536/lds_bytes, 1024/wg_size)`. |
| `register_penalty` | float | `1 + max(0, accumulators_per_sg - 8) * 0.15`. |
| `score` | float | `occupancy_proxy / register_penalty`. |
| `rank` | int | 1-indexed rank among all 642 candidates by `score` desc. |
| `shortlisted` | bool | True if in top ~24-32 by rank, or force-included as a known-measurement anchor. |
| `shortlist_reason` | string | `"top-rank"` \| `"anchor:dbuf1"` \| `"anchor:sweep-winner"` \| `"anchor:known-measurement"` \| `"known_compile_failure"` \| `"excluded"`. |

## MeasurementResult

One on-device outcome for a candidate, at a specific round.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `round` | enum | `round1_gate` \| `round2_full_shapes` \| `round3_confirmation`. |
| `correctness_status` | enum | `pass` \| `fail` \| `skipped_oversized_shape` (matches existing harness behavior for perf-only shapes). |
| `shapes_measured` | list[(K,N)] | Which production shapes this round covered. |
| `gflops_per_shape` | map[(K,N) -> float] | FLOP-weighted throughput per shape. |
| `mean_gflops` | float | Only populated for `round3_confirmation`. |
| `stddev_gflops` | float | Only populated for `round3_confirmation` (Constitution Principle IV requirement). |
| `run_count` | int | Number of independent process invocations behind `mean`/`stddev`. |
| `driver_hash` | string | The verified `vulkan.samsung.so` md5 at the time of this round (Principle VIII). |
| `clocks_pinned` | bool | Whether the pin was verified bound for this round (Principle VII). |
| `eliminated_at` | bool | True if this result caused the candidate to be dropped from the next round. |
| `elimination_reason` | string \| null | e.g. `"compile_failed"`, `"correctness_failed"`, `"below round-2 top-third cutoff"`. |

## SearchBudget

Tracks consumption against the spec's SC-001/SC-002 caps.

| Field | Type | Notes |
|---|---|---|
| `total_valid_universe` | int | 642 (fixed for this feature). |
| `configs_measured_on_hardware` | int | Distinct candidates with ≥1 `MeasurementResult`. Must stay ≤96 (SC-001). |
| `total_device_seconds` | float | Sum of on-device wall-clock time across all rounds. |
| `estimated_exhaustive_device_seconds` | float | Estimated cost of measuring all 642 at Round-2 rigor, for the SC-002 comparison. |
| `budget_exceeded` | bool | True if `configs_measured_on_hardware > 96` at any point — a hard stop condition. |

## OptimalConfiguration

The final recommended candidate (or an explicit "no improvement found"
outcome per spec FR-009).

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string \| null | Null if no candidate beat the baseline (FR-009 case). |
| `round3_result` | MeasurementResult \| null | The confirming measurement. |
| `comparison_vs_dbuf1` | float \| null | Speedup ratio, FLOP-weighted. |
| `comparison_vs_prior_winner` | float \| null | Speedup ratio vs 128×64/K16/2×2/s32, FLOP-weighted. |
| `tie_broken` | bool | Whether Decision 6's tie-breaking rule was invoked. |
| `spirv_verified` | bool | Whether the compiled SPIR-V was inspected for expected coopmat instructions (Principle VI). |
| `recommendation` | enum | `"productionize_candidate"` \| `"keep_existing_winner"`. |
