# Phase 1 Data Model: Re-Open SUBGROUP_SIZE=32 in the 8da4w CoopMat Sweep

This feature is file-based (JSON/CSV/Markdown artifacts under
`specs/026-8da4w-subgroup32-sweep/results/`), not a database-backed system. Entities below
extend `specs/025`'s data model (`ConfigurationCandidate`, `MeasurementResult`,
`SearchBudget`, `OptimalConfiguration` are carried over conceptually) with the two changes
this feature's spec requires: `subgroup_size` as a real field instead of a constant, and a
new `CorrectnessResult` entity that replaces `025`'s single-shape
`MeasurementResult.correctness_status` boolean with an explicit per-shape breakdown.

`LoopStructureResult` from `025`'s data model is reused unchanged — this feature does not
re-sweep loop structure (spec Assumptions); `dbuf2` is read from `025`'s already-recorded
result, not re-measured.

## LegalityProbeResult (new)

One User Story 1 compile/pipeline-creation attempt for a `subgroup_size=32` candidate at a
specific tile shape — the entity behind `subgroup32_legality.json`
(contracts/sweep-report-schema.md §0). Precedes `ConfigurationCandidate.compile_status` for
`subgroup_size=32` entries: `enumerate_configs.py` (Step 2) reads this file to determine
which `subgroup_size=32` candidates are even attemptable, rather than assuming legality or
illegality.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | The `tsweep_t<M>x<N>k<K>g<SGX><SGY>s32` token attempted. |
| `compile_status` | enum | `compiles` \| `compile_failed`. |
| `pipeline_creation_crashed` | bool | True only if `vkCreateComputePipelines` itself crashed (the specific historical failure mode) — distinct from a GLSL/SPIR-V compile error, which is a different `compile_failed` cause. |
| `driver_hash` | string | Verified driver identity at probe time (Principle VIII). |
| `board` | string | Which M5 EVT1 board produced this result (e.g. `xgpusw-debug08`) — this feature may use either board across different probes (spec Assumptions). |

**Derived**: the User Story 1 summary line (spec FR-002, Acceptance Scenario 1) states,
across all `LegalityProbeResult` entries, whether the historical crash reproduced at none,
some, or all attempted shapes — never generalized from a single entry.

## ConfigurationCandidate (extended)

One point in the re-derived `8da4w` tile/subgroup/subgroup-size search space, at the fixed
`dbuf2` loop structure. Unlike `025`'s version of this entity, `subgroup_size` is a real
enumerated field, not a constant.

| Field | Type | Notes |
|---|---|---|
| `wg_tile_m` | int | Output tile height. |
| `wg_tile_n` | int | Output tile width. |
| `wg_tile_k` | int | K-step per loop iteration; must divide the INT4 group size. |
| `sg_grid_x` | int | Subgroups tiling N. |
| `sg_grid_y` | int | Subgroups tiling M. |
| `subgroup_size` | int | **Changed from `025`**: `32` or `64` (research.md Decision 1) — no longer fixed. |
| `token` | string | Derived `ET_VK_DQ8CA_COOPMAT_VARIANT` value, e.g. `tsweep_t128x64k32g22s32` — the existing `tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` token format `025` already defined; this feature is the first to populate `<sub>` with `32`. |
| `wg_size` | int (derived) | `sg_grid_x * sg_grid_y * subgroup_size`. |
| `lds_bytes` | int (derived) | Same formula as `025`'s Decision 2 (`Ash_int8`/`Bsh_int8` plus broadcast arrays); independent of `subgroup_size` itself, but `wg_size` (which gates the occupancy proxy) is not. |
| `accumulators_per_sg` | int (derived) | Same as `025`: `(wg_tile_m/sg_grid_y/16) * (wg_tile_n/sg_grid_x/16)`, carried in both `int32` and `fp32` accumulator arrays. |
| `valid` | bool (derived) | Passes all constraints in Validation rules below. |
| `compile_status` | enum | `not_attempted` \| `compiles` \| `compile_failed`. Only known once actually built — **no longer assumed `compile_failed` for `subgroup_size=32` by default**, unlike `025`'s Decision 1 (this feature's whole point is to stop assuming that). |

**Validation rules**:
- `wg_size <= 1024`
- `wg_tile_m % (sg_grid_y * 16) == 0` and `wg_tile_n % (sg_grid_x * 16) == 0`
- `wg_tile_k` divides the INT4 group size
- `lds_bytes <= 65536`
- `subgroup_size ∈ {32, 64}` — **changed from `025`'s `subgroup_size == 64` hard filter**;
  both values are enumerated, and legality is determined by actual `compile_status` evidence
  (Decision 1), not assumed at enumeration time.

## CorrectnessResult (new)

Replaces `025`'s single `MeasurementResult.correctness_status` boolean with an explicit
per-shape breakdown — the entity that directly implements spec FR-003/FR-004 and closes the
gap this feature exists for.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `per_shape_results` | map[shape_id -> `pass`\|`fail`] | One entry per representative shape in the correctness harness's existing multi-shape matrix (research.md Decision 2) — must include, at minimum, every shape that appeared in either prior single-shape probe (`M=K=N=128`) plus the `M=256` family shapes this session's re-run found failing, so a regression at a previously-known-bad shape cannot silently disappear from the record. |
| `all_shapes_pass` | bool (derived) | `true` only if every entry in `per_shape_results` is `pass`. This is the field that gates ranking eligibility — **not** any single shape's result. |
| `failing_shapes` | list[shape_id] (derived) | Populated whenever `all_shapes_pass` is `false`; must be reported by name, never summarized as a bare fail count (spec FR-004). |
| `dispatch_confirmed` | bool | True only if kernel-name capture confirms the coopmat kernel dispatched (Principle VI), checked per shape if dispatch could plausibly differ by shape (e.g. small shapes falling back to tiled). |

**Derived rule**: a `ConfigurationCandidate` is eligible for the performance ranking
(`MeasurementResult`) **iff** its `CorrectnessResult.all_shapes_pass == true`. A candidate
with `all_shapes_pass == false` is recorded and reported (naming `failing_shapes`) but never
appears in the ranked performance table — this is the concrete mechanism behind spec FR-004.

## MeasurementResult (extended)

One on-device performance outcome for a `ConfigurationCandidate`, at a specific search
round. Identical in shape to `025`'s entity, with one addition:

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `ConfigurationCandidate.token`. |
| `round` | enum | `round1_gate` \| `round2_full_shapes` \| `round3_confirmation`. |
| `correctness_ref` | `CorrectnessResult` | **New in this feature**: FK to the `CorrectnessResult` that gated this candidate's entry into performance measurement — makes the traceability from FR-004 explicit rather than implicit. |
| `shapes_measured` | list[(K,N)] | Which of the 6 representative shapes (`wq`+`w1_gate` × {1B,3B,8B}) this round covered. |
| `gflops_per_shape` | map[(K,N) -> float] | FLOP-weighted throughput per shape. |
| `mean_gflops` | float | Only populated for `round3_confirmation`. |
| `stddev_gflops` | float | Only populated for `round3_confirmation` — 3-run mean, CoV<5%. |
| `run_count` | int | Must be 3 for `round3_confirmation`. |
| `driver_hash` | string | Verified driver identity at this round (Principle VIII). |
| `board` | string | **New in this feature**: which M5 EVT1 board produced this result (spec Assumptions — results may come from either board; this field makes that traceable, unlike `025` which used a single board throughout). |
| `clocks_pinned` | bool | Whether the pin was verified bound for this round (Principle VII). |
| `eliminated_at` | bool | True if this result caused the candidate to be dropped from the next round. |
| `elimination_reason` | string \| null | e.g. `"compile_failed"`, `"correctness_failed:M256_K256_N256,M256_K128_N128"` (naming the failing shapes per FR-004, not a bare category), `"below round-2 top-third cutoff"`. |

## SearchBudget (extended)

Tracks consumption against spec SC-005/FR-009 (unchanged convention from `025`: ≤15% of the
legal space, hard-capped at 30 on-device performance measurements — research.md Decision 3).

| Field | Type | Notes |
|---|---|---|
| `total_valid_universe` | int | Size of the re-derived legal space **across both subgroup sizes** — expected larger than `025`'s 542, exact count a Phase 0/1 output. |
| `budget_cap` | int (derived) | `min(round(0.15 * total_valid_universe), 30)`. |
| `configs_measured_on_hardware` | int | Distinct candidates with ≥1 `MeasurementResult` (performance stage). Correctness-only compile/gate attempts that never reach performance measurement do **not** count against this budget (same convention as `025`'s Decision 1 anchor exemption) — the cap bounds expensive performance-search device time, not the cheap correctness sweep. Must stay `<= budget_cap`. |
| `total_device_seconds` | float | Sum of on-device wall-clock time across all rounds. |
| `estimated_exhaustive_device_seconds` | float | Estimated cost of measuring all `total_valid_universe` candidates at Round-2 rigor, for an SC-006-equivalent reduction comparison if reported. |
| `budget_exceeded` | bool | True if `configs_measured_on_hardware > budget_cap` at any point — a hard stop condition. |

## OptimalConfiguration (extended)

The final recommended candidate (or an explicit "no improvement found, axis confirmed
closed" outcome per spec FR-008).

| Field | Type | Notes |
|---|---|---|
| `loop_structure` | LoopStructureResult | `025`'s already-confirmed `dbuf2` winner, read not re-measured. |
| `candidate_token` | string \| null | Null if no candidate beat `025`'s standing winner. |
| `subgroup_size_used` | int | **New in this feature**: `32` or `64` — must be stated explicitly per spec FR-007/User Story 3 Acceptance Scenario 2, regardless of which wins. |
| `round3_result` | MeasurementResult \| null | The confirming measurement (3-run mean, CoV<5%). |
| `comparison_vs_025_winner` | float \| null | Speedup ratio, FLOP-weighted, vs `025`'s standing winner (`128×32/K16/1×2/s64`, 1736 GFLOP/s) — the primary comparison this feature adds. |
| `comparison_vs_shipped_8da4w` | float \| null | Speedup ratio vs the pre-`025` shipped baseline, carried forward from `025`'s own comparison for continuity. |
| `comparison_vs_4w_winner` | float \| null | Speedup ratio vs `4w`'s winner, carried forward from `025`. |
| `tie_broken` | bool | Whether a documented tie-breaking rule was invoked. |
| `spirv_verified` | bool | Whether the compiled SPIR-V was inspected for genuine int8 cooperative-matrix instructions (Principle VI). |
| `axis_disposition` | enum | **New in this feature**: `"subgroup32_wins"` \| `"subgroup32_legal_but_no_improvement"` \| `"subgroup32_illegal_confirmed"` — the top-level answer to this feature's core question, independent of `recommendation` below. |
| `recommendation` | enum | `"productionize_candidate"` \| `"keep_025_winner"`. |
| `sg32test_probe_disposition` | string | **New in this feature**: states whether the session's ad-hoc `sg32test` shader/binding (in the `dbuf-int8-sweep` worktree) was superseded-and-removed or explicitly retained with reason (spec FR-012/SC-007). |
| `shader_comment_update` | string \| null | **New in this feature**: path to the proposed diff (or the diff itself) updating `linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s header comment per research.md Decision 6 (Principle V deliverable) — null only if genuinely not yet produced, never omitted silently. |
