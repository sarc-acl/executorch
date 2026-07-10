# Data Model: M5 EVT1 Full Microbenchmark Suite — Stable Results Report

## Entities

### Harness

One of the three existing on-device binaries.

| Field | Type | Notes |
|---|---|---|
| `name` | enum | `linear` \| `sdpa` \| `baseline` |
| `binary_name` | string | on-device filename actually invoked (e.g. `test_coopmat_linear_bench_016`) |
| `staged` | bool | confirmed present on-device before first invocation (Decision 5) |
| `raw_output_pattern` | string | which stdout lines this harness's parser looks for (e.g. linear/baseline's `SUMMARY:` table rows, SDPA's `RESULT,...` CSV lines) |

### Invocation

One full end-to-end run of a harness binary.

| Field | Type | Notes |
|---|---|---|
| `harness` | Harness | which binary |
| `rep` | int | 1, 2, or 3 (Decision 2) |
| `driver_verified` | bool | Principle VIII check passed immediately before this invocation |
| `clocks_pinned_verified` | bool | Principle VII sysfs readback passed immediately before this invocation |
| `raw_output_path` | string | `results/raw/<harness>_rep<N>.log` |
| `exit_code` | int | 0 expected; non-zero surfaced per FR-010 |

### CaseResult

One (model, scheme/op, shape, storage/regime) combination's measured
value from one Invocation. Shape of this entity differs slightly per
harness (see Key by Harness below), but every harness's case shares this
common structure.

| Field | Type | Notes |
|---|---|---|
| `harness` | Harness | which binary this case came from |
| `rep` | int | which invocation this value came from |
| `model` | string | `llama-3.2-1b` \| `llama-3.2-3b` \| `llama-3.1-8b` |
| `case_key` | string | harness-specific identity, see below |
| `tiled_value` | float | mean over the harness's own internal timed runs (us or GFLOP/s per harness's own convention) |
| `tiled_stddev` | float | the harness's own internal stddev (never dropped, per FR-007) |
| `coopmat_value` | float\|null | null for baseline bench (tiled-only harness) |
| `coopmat_stddev` | float\|null | |
| `dispatch_confirmed` | bool | from the harness's own kernel-name check (linear: `fired` bool via `!`-flag; SDPA: `dispatch_confirmed`/`NOT CONFIRMED`; baseline: N/A, always tiled by construction) |
| `correctness_status` | enum\|null | PASSED \| FAILED \| SKIPPED, where applicable |

**Key by harness**:
- `linear`: `case_key` = `(scheme, op_shape_KN)` where op_shape is one of the 4 distinct `(K,N)` shapes per model (`wq/wo`, `wk/wv`, `w1/w3`, `w2`).
- `sdpa`: `case_key` = model name alone (one case per model, no further axis).
- `baseline`: `case_key` = `(scheme, regime, storage, op)` where `op` is one of `wq/wk/wv/wo/w1_gate/w3_up/w2_down/lm_head`.

### StabilityVerdict

The cross-invocation comparison for one `(harness, model, case_key)`
tuple across its 3 (or more) Invocations' CaseResults.

| Field | Type | Notes |
|---|---|---|
| `harness` | Harness | |
| `model` | string | |
| `case_key` | string | |
| `values` | float[3] | the 3 (or N) per-invocation values being compared |
| `mean` | float | |
| `cov_pct` | float | coefficient of variation across `values`, as a percentage |
| `is_outlier` | bool | true only if `cov_pct` is a clear outlier relative to the other StabilityVerdicts' `cov_pct` within the same harness+scheme grouping (Decision 3 — no fixed cutoff) |
| `outlier_note` | string\|null | human-readable reason if `is_outlier=true` (e.g. "8.2% CoV vs <1% for every other 8da4w case in this harness") |

## Relationships

```
Harness 1--* Invocation 1--* CaseResult
(Harness, model, case_key) groups CaseResults across Invocations --> one StabilityVerdict
```

## Validation Rules

- A CaseResult with `dispatch_confirmed=false` (linear/SDPA only) MUST
  never be counted toward that scheme's "coopmat" numbers in the report
  (FR-003/FR-004) — it is reported as a tiled-fallback data point instead.
- A CaseResult with `correctness_status=FAILED` MUST be surfaced in the
  report's anomaly section (FR-010), never silently included in a
  performance table as if it passed.
- Every StabilityVerdict MUST be present in the final report for every
  case that has 3 CaseResults (SC-001) — a case missing a verdict because
  an invocation crashed is itself an anomaly to surface (FR-010), not a
  silently-dropped row.
