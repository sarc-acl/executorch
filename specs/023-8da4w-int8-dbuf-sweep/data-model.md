# Phase 1 Data Model: 8da4w Int8 WMMA Double-Buffer Variant Sweep

This feature produces measurement records and a synthesized report, not a persisted
application data model. Entities below describe the shape of the result data threaded
through the sweep (bench output → per-variant record → report), refining spec.md's Key
Entities with the concrete fields research.md's decisions imply.

## Double-Buffer Variant

One of the four loop structures under test, identified by its reference source file.

| Field | Type | Notes |
|---|---|---|
| `variant_id` | enum: `dbuf1` \| `dbuf2` \| `dbuf3` \| `dbuf4` | matches `ET_VK_DQ8CA_COOPMAT_VARIANT` values (research.md Decision 3) |
| `reference_source` | string | `.shared-context/reference-codes/shmem_double_buf{,2,3,4}.comp` |
| `shader_files` | (glsl path, yaml path) | `linear_dq8ca_q4gsw_coopmat_dbuf{N}.{glsl,yaml}` |
| `compiles` | bool | pipeline-creation succeeded without crash (User Story 1) |
| `dispatches_coopmat` | bool | bench harness's kernel-name capture confirms the int8 coopmat kernel ran, not a fallback |
| `spirv_verified` | bool | disassembly confirms genuine int8 coopmat instructions present (research.md Decision 6) |
| `correctness_passed` | bool | existing `dq8ca_q4gsw` correctness suite passed at small aligned shapes (research.md Decision 5) |
| `failure_reason` | string \| null | required whenever any of the above four booleans is false (spec FR-004); null only if all four are true |
| `timings` | list of `Timing Result` | one per shape in scope; empty if `correctness_passed` is false |

A variant with any `false` boolean and no `failure_reason` is an invalid record — the
report must never present it as measured.

## 8da4w Linear Shape

A representative `(K, N, group_size)` combination, fixed by spec Clarifications.

| Field | Type | Notes |
|---|---|---|
| `model` | enum: `1B` \| `3B` \| `8B` | LLaMA 3.2 1B / 3B, LLaMA 3.1 8B |
| `op` | enum: `wq` \| `w1_gate` | the two ops in the curated set (spec Clarifications) |
| `k` | int | contraction dim, per model (2048/3072/4096 for `wq`; same per-model `K` for `w1_gate`) |
| `n` | int | output dim, per model/op (per the existing catalog in `test_dq8ca_tile_sweep.cpp`'s `kModels`) |
| `group_size` | int | 32 for all three models (existing catalog) |
| `m` | int | fixed at `2048` (the prefill regime; matches `kM` in `test_dq8ca_tile_sweep.cpp`'s existing sweep harness) |

Six `8da4w Linear Shape` records total (2 ops x 3 models).

## Timing Result

One measurement of one variant at one shape.

| Field | Type | Notes |
|---|---|---|
| `variant_id` | enum | foreign key to Double-Buffer Variant |
| `shape` | (model, op) | foreign key to 8da4w Linear Shape |
| `mean_us` | float | 3-run mean execution time |
| `cov` | float | coefficient of variation across the 3 runs |
| `clock_pin_verified` | bool | GFLOP/s cross-check confirms the pin bound (Principle VII) |
| `driver_verified` | bool | on-device driver identity re-confirmed before this run (Principle VIII) |

A `Timing Result` is only reportable (spec SC-002) when `clock_pin_verified` and
`driver_verified` are both true.

## Sweep Report

The synthesized conclusion — one per feature run.

| Field | Type | Notes |
|---|---|---|
| `per_shape_winner` | map: shape → variant_id | fastest variant at each of the 6 shapes |
| `overall_winner` | variant_id \| `"varies by shape"` | per spec User Story 3 / Edge Cases: no forced single winner if the data doesn't support one |
| `hypothesis_verdict` | enum: `confirmed` \| `refuted` | whether dbuf3 is fastest for int8, with `Timing Result` numbers cited as evidence (spec SC-003) |
| `vs_shipped_baseline` | percentage or factor | fastest variant's measured margin over the in-sweep `dbuf4` `Timing Result`s (the shipped production loop structure, measured under this same harness for an apples-to-apples comparison — spec SC-004) |
| `failed_variants` | list of `Double-Buffer Variant` (where any boolean is false) | included explicitly, with `failure_reason`, per spec FR-004/SC-001 |

## Relationships

```text
Double-Buffer Variant (1) ──< Timing Result >── (1) 8da4w Linear Shape
Sweep Report ──references──> Double-Buffer Variant, Timing Result
```

No entity has a lifecycle/state-transition beyond the boolean gates above: a variant is
either fully valid (all four booleans true, timings present) or explicitly failed
(`failure_reason` set, no timings) — there is no intermediate or mutable state to model.
