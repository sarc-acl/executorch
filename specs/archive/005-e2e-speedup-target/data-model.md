# Data Model: End-to-End Speedup Target and Validation

## Speedup Target

One entry per (model, scheme) — 6 total. Produced now, directly from `001`'s
existing baseline data (real, not synthetic).

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | One of the six `001`-`004` configurations |
| `baseline_prefill_tokens_per_sec` | float | Copied verbatim from `001`'s `e2e.prefill_tokens_per_sec` |
| `baseline_prefill_stdev` | float | Copied from `001`'s `e2e.variance.prefill_tokens_per_sec_stdev` — the noise band used for the met/exceeded boundary |
| `target_multiplier` | float | `2.0`, fixed per the Clarifications session |
| `target_prefill_tokens_per_sec` | float | `baseline_prefill_tokens_per_sec * target_multiplier` |
| `baseline_source` | string | Path to the exact `001` JSON file this was read from, for traceability |

## Re-Measurement

One entry per (model, scheme), produced **later** by a future feature once a
build with real optimization work exists. Same schema as `001`'s `e2e`
object (Decision 1) — this feature does not define a new one.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | |
| `e2e` | object | Identical shape to `001`'s baseline `e2e` object: `prefill_tokens_per_sec`, `decode_tokens_per_sec`, `prefill_tokens`, `decode_tokens`, `num_runs`, `variance`, `run_metadata` |
| `methodology_comparable` | bool | `true` unless device, workload size, or statistical discipline diverged from the baseline's (FR-008) |
| `methodology_note` | string or null | Required when `methodology_comparable` is `false` — states what diverged |

## Outcome

One entry per (model, scheme) — the result of comparing a Re-Measurement
against its Speedup Target.

| Field | Type | Notes |
|---|---|---|
| `model` / `scheme` | string | |
| `observed_multiplier` | float or null | `after.e2e.prefill_tokens_per_sec / baseline_prefill_tokens_per_sec`; null if `methodology_comparable` is false |
| `verdict` | enum | `met` / `exceeded` / `missed` / `regressed` / `not_comparable` (Research Decision 4's thresholds) |
| `combined_e2e_change_pct` | float or null | Tracked, reported change in combined e2e tok/s (weighted by the fixed 2048/1024 workload) — **never** used to compute `verdict`, per FR-001/FR-006 |
| `is_synthetic` | bool | `true` for every self-test entry; `false` only for a real re-measurement. Rendered prominently in any report so synthetic and real results can never be confused. |

## Outcome Report

The rendered document: one section per (model, scheme) showing baseline,
target, actual re-measured prefill number, verdict, and the tracked combined
e2e change — plus a top-level summary a reader can scan without opening any
per-config detail. Two variants exist, kept in physically separate files
(Research Decision 3): `results/selftest/selftest-outcome-report.md`
(`is_synthetic: true` throughout, produced now) and the future real
`results/outcome-report.md` (`is_synthetic: false` throughout, produced once
real re-measurement data exists).

No lifecycle/state transitions beyond: Speedup Target exists now → Speedup
Target + Re-Measurement (real or synthetic) → Outcome is computed from both.
