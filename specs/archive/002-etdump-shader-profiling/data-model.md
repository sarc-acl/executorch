# Data Model: ETDump E2E Shader Profiling Breakdown

Entities mirror the spec's Key Entities section (as refined by the
Clarifications session), made concrete for the parsing script and report
format. This feature is analysis-only — all of it is flat-file data derived
from `.etdump` captures plus `001`'s existing `results/shapes.json`.

## Kernel Invocation (raw)

One event, straight off the wire from a single `.etdump` file — the
companion data referenced in the Clarifications answer.

| Field | Type | Notes |
|---|---|---|
| `config` | (model, scheme) ref | Which of the six `001` configurations this came from |
| `phase` | enum | `prefill` or `decode` |
| `kernel_name` | string | The Vulkan shader/kernel name embedded in the event (e.g. `linear_q4gsw_tiled_...`) |
| `operator_name` | string | The higher-level op the kernel implements (e.g. `et_vk.linear_q4gsw.default`), if present in the embedded JSON |
| `shape` | {m,k,n} or null | Parsed from the event's embedded arg `"sizes"` JSON (Research Decision 2); null for non-matmul ops |
| `time_us` | float | This single dispatch's GPU time, from the event's timestamp pair |
| `sequence_index` | int | Position within the phase's event stream (lets someone reconstruct per-layer order if needed) |

## Aggregated Kernel Entry

The primary breakdown row — one per unique `(kernel_name, shape)` pair
within a phase, per the Clarifications answer.

| Field | Type | Notes |
|---|---|---|
| `config` | (model, scheme) ref | |
| `phase` | enum | `prefill` or `decode` |
| `kernel_name` | string | |
| `shape` | {m,k,n} or null | Not-applicable for non-matmul kernels (FR-003) |
| `total_time_us` | float | Sum of `time_us` across all Kernel Invocations with this (kernel_name, shape) in this phase |
| `invocation_count` | int | How many raw invocations were aggregated |
| `pct_of_phase` | float | `total_time_us / phase_wall_clock_us_profiled` (see Profiling Run) |
| `category` | string | Assigned per Research Decision 4 |

## Profiling Run

One (model, scheme, phase) capture.

| Field | Type | Notes |
|---|---|---|
| `config` | (model, scheme) ref | |
| `phase` | enum | `prefill` or `decode` |
| `device` | string | Fixed: `rocky-ryzen` |
| `dispatch_path` | string | Fixed: `tiled_baseline`, inherited from `001` |
| `etdump_path` | path | Location of the raw `.etdump` file |
| `phase_wall_clock_us_profiled` | float | This phase's total time *as measured during this profiled run* (FR-006) — the percentage denominator |
| `phase_wall_clock_us_baseline` | float | The corresponding un-profiled number from `001`'s `results/raw/<model>_<scheme>.json`, shown alongside for overhead comparison (FR-005) |
| `attributed_pct` | float | `sum(total_time_us across all Aggregated Kernel Entries) / phase_wall_clock_us_profiled` — the reconciliation number (FR-005); any gap is unattributed overhead, reported honestly rather than hidden |
| `decode_window_steps` | int or null | Number of decode steps profiled (null for prefill) — see Research Decision 5 |

## Category Rollup

A named grouping derived from a Profiling Run's Aggregated Kernel Entries.

| Field | Type | Notes |
|---|---|---|
| `config` | (model, scheme) ref | |
| `phase` | enum | |
| `category` | string | One of: attention projection, feed-forward, output/vocab projection, non-shader overhead, other |
| `total_time_us` | float | Sum of `total_time_us` across Aggregated Kernel Entries in this category |
| `pct_of_phase` | float | |

Categories' `pct_of_phase` values, plus any unattributed remainder
(`1 - attributed_pct`, reported as an explicit "unattributed" row rather
than silently dropped), sum to 100% of the phase.

No lifecycle/state transitions apply — each entity is written once by the
parsing script and read thereafter by `profiling-report.md` and by anyone
doing follow-on analysis.
