# Data Model: WMMA-Optimizable Shader Candidates Report

Entities mirror the spec's Key Entities section. This feature is analysis-only
— all of it is derived from `002`'s already-existing JSON plus citations into
already-existing source code; nothing here is captured live.

## Shader Classification

One row per (configuration, phase, kernel_name, shape) — the same grain as
`002`'s Aggregated Kernel Entry, extended with WMMA-candidacy fields.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | Which of the six `001`/`002` configurations |
| `phase` | enum | `prefill` or `decode` |
| `kernel_name` | string | Carried over from `002` |
| `shape` | {m,k,n} or null | Carried over from `002` |
| `category` | string | Carried over from `002`'s category rollup |
| `classification` | enum | `a` (WMMA in effect), `b` (WMMA exists, blocked), `c` (no WMMA implementation), `d` (not applicable), `uncertain` |
| `blocking_reasons` | list[string] | Populated for `b`/`c`; **may have more than one entry** (e.g. the prefill linear family has two independent reasons — see research.md) |
| `existing_or_prospective_shader` | string or null | The named coopmat shader that already implements this (classification `a`/`b`) or would need to (`c`) |
| `total_time_us` | float | Carried over from `002`'s `aggregated[].total_time_us` |
| `pct_of_phase` | float | Carried over from `002`'s `aggregated[].pct_of_phase` |

## Optimization Candidate Group

The rolled-up entity behind the ranked report (US3). One per distinct root
cause (not per config) — see research.md Decision 3 for the four groups
found in practice.

| Field | Type | Notes |
|---|---|---|
| `group_name` | string | e.g. "Prefill linear GEMM -- blocked by rank-3 output + TEXTURE_3D storage" |
| `classification` | enum | `b` or `c` (only classifications promoted to candidates) |
| `blocking_reasons` | list[string] | The shared reason(s) across every member row |
| `existing_or_prospective_shader` | string | The shader family this group refers to |
| `member_rows` | list[Shader Classification ref] | Every row across all six configs/phases contributing to this group |
| `total_time_us_summed` | float | Sum of `total_time_us` across all `member_rows` -- the primary sort key (Clarifications: absolute time) |
| `per_config_breakdown` | list[{model, scheme, phase, total_time_us, pct_of_phase}] | Supporting detail so a reader can see which configs/phases contribute how much |

## Candidates Report

The consolidated markdown document: Optimization Candidate Groups ranked by
`total_time_us_summed` descending, split into two clearly-labeled sections
("existing implementation blocked" vs. "no implementation yet"), each group
showing its per-config breakdown table. References `002`'s
`profiling-report.md` and each config's raw JSON rather than duplicating
their content.

No lifecycle/state transitions apply — every entity here is written once by
the classification script and read thereafter by the report.
