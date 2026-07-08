# Data Model: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

## WMMA-Eligible Export

One entry per (model, scheme) -- 6 total (3 models x 2 int4 schemes, the
constitution's default scope).

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | e.g. `llama-3.2-1b`/`4w` |
| `rank3_fix_applicable` | bool | Whether research.md Decision 1's guard relaxation is expected to apply (always true for this workstream's exported models -- batch is always 1 per `003` -- but recorded per-config rather than assumed, per FR-007) |
| `export_status` | enum | `exported` / `blocked` (rank-3 fix could not be applied for this configuration, or export itself failed -- FR-007). A `blocked` entry MUST carry a reason string |
| `smoke_check_status` | enum | `pass` / `fail` / `not_run` -- `006`'s coherent/non-degenerate output bar. No dispatch check or timing is attempted unless this is `pass` |
| `dispatch_kernel_wq` / `dispatch_kernel_w1_gate` (or similar per-op sample) | string | Kernel name(s) read from the ETDump capture for this export's linear ops (research.md Decision 4) |
| `dispatch_status` | enum | `confirmed` (kernel name contains `_coopmat`) / `fallback` (it doesn't) / `not_run`. FR-003: no WMMA tok/s number is reported unless `confirmed` |
| `correctness_status` | enum | `verified` / `not_verified` -- whether research.md Decision 2's new rank-3 correctness check passed for this configuration's scheme (`4w` vs `8da4w` each need their own check, since they're different shaders) |

## E2E Three-Way Comparison Case

One entry per (model, scheme, phase) -- 6 configurations x 2 phases
(prefill, decode) = 12 rows, the report's core comparison unit.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` / `phase` | string | e.g. `llama-3.2-1b`/`4w`/`prefill` |
| `texture3d_tok_s` / `texture3d_stdev` | float | From `006`'s report, reused verbatim (research.md Decision 5) -- not re-captured |
| `buffer_tok_s` / `buffer_stdev` | float | From `006`'s report, reused verbatim -- not re-captured |
| `wmma_tok_s` / `wmma_stdev` / `wmma_num_runs` | float / float / int | This feature's own new capture (research.md Decision 6). Absent (not populated) unless the parent WMMA-Eligible Export's `dispatch_status == confirmed` |
| `wmma_vs_buffer_pct` / `wmma_vs_texture3d_pct` | float | `(buffer_tok_s - wmma_tok_s) / buffer_tok_s * 100` and the Texture3D equivalent; positive means WMMA is slower |
| `cross_session_caveat` | bool | `true` for every `prefill` row (research.md Decision 5's inherited `006` caveat), `false` for every `decode` row |
| `microbench_consistency` | enum | `consistent` / `diverges` / `not_applicable` (when `wmma_tok_s` is absent) -- per research.md Decision 7, compares this row's e2e direction against `007`'s microbenchmark-level finding for that scheme, and for `8da4w` additionally against `008`'s shipped-config finding |

## E2E tok/s Report

The consolidated document (US3): per-scheme summary statements at the top
(`4w` and `8da4w` each get their own "does WMMA help" verdict -- research.md
Decision 7 -- never blended into one number), followed by the 12-row
comparison table (or fewer, if some configurations are blocked), followed by
a Blocked/Failed section listing every WMMA-Eligible Export whose
`export_status`, `smoke_check_status`, or `dispatch_status` prevented a
measurement, each with its specific reason (FR-007, SC-004) -- present even
if empty, matching `006`/`007`'s convention.

No lifecycle/state transitions -- this is a one-shot capture-and-compare,
same shape as `006`'s own report.
