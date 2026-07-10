# Data Model: Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics

## Entities

### UnifiedResultLine

The single record format all three harnesses emit, one per completed
case, printed immediately (research.md Decision 1).

| Field | Type | Notes |
|---|---|---|
| `harness` | enum | `linear` \| `sdpa` \| `baseline` |
| `model` | string | `llama-3.2-1b` \| `llama-3.2-3b` \| `llama-3.1-8b` |
| `scheme` | string | `4w` \| `8da4w` (baseline/linear); N/A for SDPA (no quantization scheme axis) |
| `regime` | enum | `prefill` \| `decode` — new first-class axis for linear and SDPA (baseline already had it) |
| `variant` | enum | `tiled` \| `coopmat` (linear/baseline); `qk` \| `av` \| `total` (SDPA) |
| `k`, `n` | int\|null | shape dims where applicable (null for SDPA, which uses `head_dim`/`num_heads`/`num_kv_heads` instead — carried in a harness-specific extra field, not part of the shared schema's required fields) |
| `avg_us` | float | mean over the harness's own internal timed runs |
| `stddev_us` | float | the harness's own internal stddev — never dropped (FR unchanged from `specs/020`) |
| `gflops` | float\|`-1` | `-1` sentinel where GFLOP/s isn't the primary metric (SDPA) |
| `dispatch_status` | enum | `confirmed` \| `fallback_tiled` \| `not_applicable` (research.md Decision 2) |
| `correctness_status` | enum | `PASSED` \| `FAILED` \| `SKIPPED` \| `CRASHED` (baseline bench only, added per research.md Decision 9 -- a case-local exception, e.g. the `lm_head` QueryPool race, with `avg_us`/`stddev_us`/`gflops` all `-1` sentinels) |

### RegimeCase (per-harness case definition, post-redesign)

| Harness | Regimes | Variants | New in this feature |
|---|---|---|---|
| linear | prefill(M=2048), decode(M=1) | tiled, coopmat | both regimes now real (was M=1024 only) |
| sdpa | prefill(S=2048), decode(S=1, input_pos=3071) | qk, av, total | decode case + qk/av split (was 1 combined prefill row) |
| baseline | prefill(M=2048), decode(M=1) | tiled@texture3d, tiled@buffer | unchanged (already had both regimes); execution now one `execute_test_cases()` call per case |

### ExecutionBatch (baseline bench only — organizational, not the memory-safety mechanism)

| Field | Type | Notes |
|---|---|---|
| `model` | string | one of the 3 models — an output-grouping unit, not a shared `execute_test_cases()` call |
| `case_count` | int | 64 cases per model, each run via its own individual `execute_test_cases()` call (research.md Decision 8) |
| `peak_memory_estimate_gb` | float | ~0.5GB worst case (one `lm_head` prefill case) — bounded by per-case execution, not by this grouping |

## Relationships

```
Harness 1--* RegimeCase 1--* UnifiedResultLine
ExecutionBatch (baseline only) 1--* UnifiedResultLine (one model's worth of cases, each its own execute_test_cases() call)
```

## Validation Rules

- Every `regime=decode` `UnifiedResultLine` (both linear and SDPA) MUST
  have `dispatch_status = not_applicable` — both harnesses' decode cases
  hit an explicit `is_gemv`-style short-circuit (`QuantizedLinear.cpp`'s
  `is_gemv_case`, `SDPA.cpp`'s `is_gemv`) that dispatches a dedicated
  `_coop` kernel before the coopmat eligibility check ever runs, so
  neither `confirmed` nor `fallback_tiled` is ever correct for a decode
  case — a `confirmed` decode result indicates a bug in the harness's own
  status-derivation logic, not a real coopmat win (spec.md Edge Cases).
- Every baseline `UnifiedResultLine` MUST have
  `dispatch_status = not_applicable` regardless of regime — baseline has
  no coopmat toggle at all (research.md Decision 2).
- A harness crash mid-batch MUST NOT invalidate `UnifiedResultLine`s
  already printed by prior, completed batches/cases (spec.md Edge Cases)
  — this is a property of print-immediately (Decision 1) plus
  per-model batching (Decision 3), not something the parser needs to
  separately enforce.
