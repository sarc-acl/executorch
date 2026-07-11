# Data Model: MiniPC RDNA3 Handoff Report

## Consolidated Finding

One entry per spec (`001`-`012`).

| Field | Notes |
|---|---|
| `spec` | e.g. `009-e2e-tokrate-report` |
| `headline` | The one-line result (research.md Decision 1) |
| `tier` | `microbenchmark` / `e2e` / `n/a` (e.g. `003`'s candidate survey, `005`'s target-setting) |
| `source_file` | The exact results file the headline is cited from (FR-002 traceability). N/A for goal-setting entries with no results file (`005` -- no measured result to cite) |

## Repo Handoff State

Single instance, not per-spec.

| Field | Notes |
|---|---|
| `branch` | `quant-perf-optimization` |
| `last_commit` | `d8800fb02e` |
| `uncommitted_file_count` | 71 (research.md Decision 3) |
| `uncommitted_scope` | Specs `007`-`013` and their underlying production-code fixes |
| `prerequisite` | Commit/push, explicitly named, not performed by this feature |

## Samsung/Xclipse Runbook Item

One entry per step of the established methodology.

| Field | Notes |
|---|---|
| `step` | export / build / dispatch-confirm / benchmark / report |
| `status` | `carries_over_unchanged` / `needs_adaptation` / `newly_established` |
| `check_first` | Populated only for `newly_established` steps -- e.g. Xclipse coopmat support + tile dimensions before the first benchmark |

## Handoff Report

The consolidated document: Consolidated Findings table, open items
(research.md Decision 2), Repo Handoff State, Samsung/Xclipse Runbook.
One-shot, no lifecycle -- matches this workstream's established report
shape.
