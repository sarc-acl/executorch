# Data Model: M5 EVT1 `4w` Linear Coopmat Retune

## Retuned Shader Change

One of the three code changes already applied to `linear_qw_coopmat.glsl`.

| Field | Type | Notes |
|---|---|---|
| `name` | string | `fp16_accumulate` / `loop_flattening` / `vectorized_dequant` |
| `risk_level` | enum | `precision_risk` (fp16_accumulate only) / `same_math_code_shape` (the other two) |
| `origin` | string | Where the change came from -- fp16_accumulate: new experiment, not previously measured anywhere; loop_flattening: dbuf1 variant algorithm already chosen as the sweep winner in specs `007`-`012` (MiniPC), re-expressed in this flattened code shape; vectorized_dequant: pure ALU-reduction rewrite, no prior sweep |
| `correctness_gate` | string | Reference to `test_coopmat_linear_bench.cpp`'s correctness harness AS EXTENDED by FR-008 (`research.md` Decision 2, revised) -- run at production K=2048/4096; the pre-extension harness (K<=256 only) does not satisfy this gate |
| `correctness_result` | enum | `pass` / `fail` / `not_yet_run` |
| `perf_baseline` | string | Reference to Decision 1's fresh pre-change M5 EVT1 build -- the same baseline for all three changes |
| `perf_result` | record | `{mean_us, stdev_us, iterations, kernel_dispatched, spirv_verified}` or `not_yet_run` |
| `disposition` | enum | `keep` / `keep_with_caveat` / `revert` / `pending` |
| `disposition_reason` | string | Free text -- required whenever `disposition != pending` |

Seeded rows (as of this feature's start, before any hardware run):

| name | risk_level | correctness_result | perf_result | disposition |
|---|---|---|---|---|
| `fp16_accumulate` | precision_risk | not_yet_run | not_yet_run | pending |
| `loop_flattening` | same_math_code_shape | not_yet_run | not_yet_run | pending |
| `vectorized_dequant` | same_math_code_shape | not_yet_run | not_yet_run | pending |

## Documentation Clarification

The one non-code-behavior change (item 4).

| Field | Type | Notes |
|---|---|---|
| `files` | list | `linear_dq8ca_qw_coopmat.glsl`, `QuantizedLinear.cpp` (`add_linear_dqa_qw_node`) |
| `finding_date` | date | 2026-06-30 |
| `finding` | string | UBO-direct loop-bound/store-width method produces wrong results on this shader at M>=128; spec-const workaround must not be dropped despite the underlying Xclipse driver bugs being otherwise fixed |
| `validation_gate` | string | None -- ships unconditionally per spec Clarifications (comment-only, no runtime effect) |
| `disposition` | enum | Fixed at `keep` (not subject to the `pending`/hardware-gated lifecycle above) |

## Correctness Harness Extension (FR-008)

A prerequisite, not a per-change entity: the set of new
`kCorrectnessShapes`/`kRank3CorrectnessShapes` entries added to
`test_coopmat_linear_bench.cpp` before any Validation Result below can be
produced.

| Field | Type | Notes |
|---|---|---|
| `shapes_added` | list | e.g. `[{M:128,K:2048,N:128}, {M:128,K:4096,N:128}]` -- at minimum K=2048/4096, `M`/`N` chosen coopmat-eligible (`M%64==0`, `N%64==0`, `K%32==0`) per existing gate rules |
| `data_strategy` | string | Reused unchanged from the existing K<=256 cases: deterministic, well-conditioned (positive-only) activations/weights -- no new data-generation logic |
| `tolerance` | string | Reused unchanged: `abs=0.5`, `rel=0.05` |
| `status` | enum | `not_yet_written` / `written` / `passing` / `failing` |

## Validation Result

One correctness-or-performance outcome record for one Retuned Shader Change,
produced by User Story 2 or 3.

| Field | Type | Notes |
|---|---|---|
| `change_name` | string | FK to Retuned Shader Change |
| `tier` | enum | `correctness` / `tier1_microbench` |
| `shapes_tested` | list | e.g. `[K=2048, K=4096]` for correctness; real per-model prefill shapes for tier-1 |
| `kernel_dispatched` | string | Actual kernel name observed -- must confirm coopmat, not a tiled fallback (Principle VI) |
| `spirv_verified` | bool | Per `research.md` Decision 4 |
| `numerical_tolerance` | string | Only populated for `fp16_accumulate`'s correctness tier -- fixed at `abs=0.5`/`rel=0.05`, `test_coopmat_linear_bench.cpp`'s existing well-conditioned-data tolerance (per spec FR-004/FR-008), not a newly-invented value |
| `outcome` | enum | `pass` / `fail` |
| `notes` | string | Free text, e.g. divergence magnitude on failure |

## Lifecycle

```
pending --(correctness check run)--> correctness pass/fail
  correctness fail --> disposition = revert (fp16_accumulate) OR keep_with_caveat (if a same-math change somehow fails, which would indicate a bug in the rewrite itself, not a precision tradeoff)
  correctness pass --> (perf run, if pursued) --> disposition = keep | keep_with_caveat
blocked_on_hardware_access --> disposition remains "pending", explicitly labeled per spec FR-006
```

No other state transitions -- this is a one-shot validate-and-decide
feature per change, not a recurring or multi-round process.
