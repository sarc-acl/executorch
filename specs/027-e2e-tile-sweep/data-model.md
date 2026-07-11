# Phase 1 Data Model: 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

This feature is file-based (JSON/Markdown artifacts under
`specs/027-e2e-tile-sweep/results/`), not a database-backed system.

## Candidate (merged pre-filter entry)

One tile/subgroup/subgroup-size configuration already correctness-verified by `025` or
`026`, carrying its existing microbenchmark score — this feature's input, not something it
re-derives.

| Field | Type | Notes |
|---|---|---|
| `token` | string | The existing `tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` dispatch token (or the shipped-baseline sentinel, see `BASELINE_TOKEN` below). |
| `source_feature` | enum | `"025"` \| `"026"` — which prior feature's data this candidate comes from. |
| `subgroup_size` | int | `32` or `64`. |
| `microbenchmark_gflops` | float | The existing measured GFLOP/s from `025`'s Round 3 or `026`'s Round 3 confirmation. |
| `microbenchmark_rank` | int | This candidate's rank in the *combined* `025`+`026` ranking (research.md Decision 1) — 1-indexed, ties broken by `source_feature` recency (`026` before `025`) then token. |
| `correctness_all_shapes_pass` | bool | Must be `true` to be eligible for this feature's shortlist (spec FR-004) — read directly from `025`'s per-candidate correctness gate or `026`'s `correctness_matrix.json`. |
| `shape_family` | enum | `"1B"` \| `"3B"` \| `"8B"` — which model size's per-layer shapes this candidate's `microbenchmark_gflops` was measured at (research.md Decision 2). For every candidate in this feature's initial 8-candidate shortlist, this is `"8B"` (both `025` and `026` measured at the 8B-shaped `K=4096/14336` representative shape). |
| `model_used` | string | Resolved PTE filename for e2e measurement, derived from `shape_family` (e.g. `llama3_1_8b_8da4w_buffer_ctx3072.pte` for `"8B"`) — never hardcoded independent of `shape_family`. |
| `shortlisted` | bool | `true` for the top 8 by `microbenchmark_rank` among `correctness_all_shapes_pass: true` candidates (spec Clarifications). |

`BASELINE_TOKEN = "(unset — default dispatch)"`: the currently-shipped configuration is
represented as a `Candidate` with no `ET_VK_DQ8CA_COOPMAT_VARIANT` value set, so it can be
compared using the same `E2EMeasurement` shape as every swept candidate, rather than as a
special-cased number.

## E2EMeasurement

One real end-to-end run for a `Candidate` on its `model_used` PTE.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `Candidate.token` (or `BASELINE_TOKEN`). |
| `stage` | enum | `"screen"` (1 run) \| `"confirm"` (3 fresh runs, research.md Decision 4). |
| `run_index` | int | 1-indexed within its `stage` (always `1` for `screen`; `1`-`3` for `confirm`). |
| `prefill_tok_s` | float | Read directly from the runner's `PyTorchObserver` JSON line — never estimated (Principle VI). |
| `decode_tok_s` | float \| null | Recorded opportunistically (spec Assumptions: prefill is the primary metric). |
| `model_used` | string | Echoed from `Candidate.model_used` for this measurement — a mismatch between this field and the candidate's own `shape_family` is a contract violation (research.md Decision 2). |
| `driver_hash` | string | Verified driver identity at measurement time (Principle VIII). |
| `board` | string | Which M5 EVT1 board produced this result. |
| `clocks_pinned` | bool | Whether the pin was verified bound (Principle VII). |
| `coherence_checked` | bool | Whether a short-prompt sanity check (coherent, non-garbage output) was run for this `candidate_token`+`model_used` combination before trusting its timing numbers — required once per combination, not per run. |

**Derived**: `screen_ratio(candidate)` = `(mean(screen prefill_tok_s) - mean(baseline screen prefill_tok_s)) / mean(baseline screen prefill_tok_s)`. `escalate_to_confirm(candidate)` = `screen_ratio(candidate) >= -0.10` (research.md Decision 3).

## ConfirmationResult

The 3-run statistical summary for a candidate that screened within 10% of, or ahead of,
baseline (research.md Decision 3).

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `Candidate.token`. |
| `mean_prefill_tok_s` | float | Mean of the 3 `confirm`-stage `E2EMeasurement.prefill_tok_s` values. |
| `stddev_prefill_tok_s` | float | Stddev of the same 3 values. |
| `cov` | float | `stddev / mean`. |
| `baseline_mean_prefill_tok_s` | float | The shipped baseline's own confirmed 3-run mean (baseline is always confirmed, regardless of any screen-ratio threshold — spec FR-011: the comparison target itself must be real, multi-run evidence, not a single number). |
| `improvement_pct` | float | `(mean_prefill_tok_s - baseline_mean_prefill_tok_s) / baseline_mean_prefill_tok_s * 100`. |
| `beats_baseline` | bool (derived) | `true` only if `improvement_pct > 0` AND the two 3-run ranges (mean ± stddev) don't overlap — the same non-overlapping-distributions bar this session's own `026` Tier-2 check already used as its practical confidence signal. |

## SearchExtensionBudget (User Story 2, conditional)

Only populated if User Story 1's shortlist produces zero `beats_baseline: true`
`ConfirmationResult`.

| Field | Type | Notes |
|---|---|---|
| `triggered` | bool | Whether User Story 2 ran at all (spec FR-006). |
| `new_candidates_selected` | int | Count of genuinely new tile/subgroup/subgroup-size combinations built and measured (0 if not triggered). |
| `selection_rationale` | string | Cites the existing `025`/`026` analytical scoring model plus what User Story 1 found about microbenchmark-vs-e2e rank agreement (spec FR-007) — never "arbitrary" or absent if `triggered: true`. |
| `budget_cap` | int | The small, pre-declared cap this extension may not exceed (spec FR-007/FR-009). |

## FinalAnswer

This feature's single, mandatory deliverable (spec FR-008/SC-001).

| Field | Type | Notes |
|---|---|---|
| `winner_token` | string | `BASELINE_TOKEN` if no candidate beat baseline; otherwise the winning `Candidate.token`. |
| `winner_confirmation` | ConfirmationResult \| null | Null only if `winner_token == BASELINE_TOKEN`. |
| `microbenchmark_vs_e2e_rank_agreement` | enum | `"agree"` \| `"partially_agree"` \| `"disagree"` — spec SC-006, stated regardless of which token wins. |
| `candidates_measured_e2e` | int | Total distinct candidates taken to at least a screening run (User Story 1 + any User Story 2 extension). |
| `candidates_skipped_reasons` | map[token -> string] | Every shortlist-eligible candidate NOT measured, with why (spec SC-005) — e.g. a `025`/`026` candidate that was correctness-verified but ranked below the top-8 cutoff. |
