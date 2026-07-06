# Data Model: WMMA Coopmat Improvement Microbenchmark

## WMMA Comparison Case

One entry per (model, scheme, op) -- 42 total (3 models x 2 schemes x 7 ops;
`lm_head` excluded per `research.md` Decision 3), each carrying both the
tiled-baseline and WMMA measurements for the prefill (`M=2048`), Buffer-storage
shape.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` / `op` | string | e.g. `llama-3.2-1b`/`4w`/`w1_gate`. Regime is always `prefill`; storage is always `buffer` (fixed by scope, not a variable dimension in this feature) |
| `m` / `k` / `n` | int | Shape, from `001`'s `results/shapes.json` (duplicated in the harness's `kModels`) |
| `tiled_mean_us` / `tiled_stdev_us` | float | From `004`'s existing `storage_bench_raw.log`, `storage=buffer`, `regime=prefill` row (Decision 2) -- not re-captured |
| `wmma_mean_us` / `wmma_stdev_us` | float | From this feature's new capture: the same harness run with `ET_VK_FORCE_TILED_LINEAR` unset |
| `tiled_kernel` / `wmma_kernel` | string | Captured kernel names. `tiled_kernel` MUST be from the tiled/coop family (never `coopmat`); `wmma_kernel` MUST contain `coopmat` -- a case violating either is not a valid comparison (see `dispatch_status` below) |
| `dispatch_status` | enum | `confirmed` (`wmma_kernel` contains `coopmat`) / `fallback` (it doesn't -- tile-alignment or other eligibility check unexpectedly failed; FR-004) |
| `spirv_verified` | bool | Whether `research.md` Decision 4's SPIR-V inspection confirmed `OpCooperativeMatrix*KHR` instructions for this case's kernel (checked once per distinct kernel name, not per case, since the same compiled shader serves every shape) |
| `correctness_verified` | bool | `dispatch_status == confirmed AND spirv_verified AND` the kernel family is covered by `test_coopmat_linear_bench.cpp`'s existing `kCorrectnessShapes` (Decision 7) |
| `speedup_pct` | float | `(tiled_mean_us - wmma_mean_us) / tiled_mean_us * 100` -- positive means WMMA is faster |
| `significance` | enum | `real_effect` / `noise`, via the same non-overlapping `mean +/- 2*stdev` band rule `004` already established |
| `weight` | float | This op's share of its configuration's 7 measured ops' total tiled-baseline time (`tiled_mean_us` / sum of all 7 ops' `tiled_mean_us` for that model/scheme) -- the weight used in the overall figure. Revised from the original plan to use `003`'s `pct_of_phase` directly: that data is aggregated by `(kernel_name, shape)`, not per named op, and cannot be cleanly split for same-shape sibling pairs (`wq`/`wo`, `wk`/`wv`, `w1_gate`/`w3_up`) without an invented assumption (research.md Decision 6 addendum) |

## Excluded / Out-of-Scope Entry

A separate list, always rendered (even if empty), covering:
- `lm_head` for every (model, scheme) -- excluded per Decision 3, with the
  stated reason.
- Decode-regime GEMV ops -- out of scope per FR-006, with the stated reason.
- Any case where `dispatch_status == fallback` or `correctness_verified ==
  false` -- excluded from the main comparison table and the time-weighted
  overall figure, listed here instead with the specific reason (which check
  failed).

## WMMA Improvement Report

The consolidated document (US3): one time-weighted overall improvement
figure (Decision 6) at the top, followed by the full 42-row case table
sorted by `model`, `scheme`, `op`, followed by the Excluded/Out-of-Scope
section per FR-009 -- never silently dropped from the configuration count.

No lifecycle/state transitions -- this is a one-shot capture-and-compare,
same shape as `001`/`004`'s microbenchmark reports.
