# Phase 1 Data Model: 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

This feature is file-based (JSON/Markdown artifacts under
`specs/028-4w-e2e-tile-sweep/results/`), not a database-backed system.

## PortVerification (new — this feature's own prerequisite, `027` had no equivalent)

Records that the one-time tsweep infra port (research.md Decision 0) preserved
correctness before any e2e measurement is allowed to trust it.

| Field | Type | Notes |
|---|---|---|
| `port_commit` | string | Commit hash on the `028-4w-e2e-tile-sweep` execution branch that lands the ported `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` + `ET_VK_Q4GSW_COOPMAT_VARIANT` dispatch token. |
| `base_shader_reference` | string | `linear_q4gsw_coop.glsl` (dev's current shader used as the port's structural base — research.md Decision 0). |
| `archived_patch_reference` | string | Path to the read-only reference patch (`.archived-artifacts/tmp-origcm-2026-07-08/...`) — cited, never applied directly. |
| `candidate_token` | string | FK to `Candidate.token` — one `PortVerification` entry per shortlisted candidate. |
| `correctness_status` | enum | `"pass"` \| `"fail"` — result of re-running `022`'s existing fp32-reference correctness check (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) against the ported shader for this token. |

**Contract**: a `Candidate` MUST NOT proceed to any `E2EMeasurement` (screen or confirm)
unless its `PortVerification.correctness_status == "pass"` (spec FR-004, extended by
plan.md Technical Context's Testing section to cover the port itself, not just `022`'s
original measurement).

## Candidate (merged pre-filter entry)

One tile/subgroup configuration already correctness-verified by `022`, carrying its
existing microbenchmark score — this feature's input, not something it re-derives.

| Field | Type | Notes |
|---|---|---|
| `token` | string | The existing `tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` dispatch token (or the shipped-baseline sentinel, see `BASELINE_TOKEN` below). |
| `source_feature` | string | `"022"` — every shortlist candidate in this feature comes from one source (research.md Decision 1), unlike `027`'s two-source (`025`/`026`) merge. |
| `microbenchmark_gflops` | float | The existing measured `mean_gflops`/`gflops` from `022`'s `round2_results.json`. |
| `microbenchmark_rank` | int | 1-indexed rank by `microbenchmark_gflops` descending among `022`'s Round 2 candidates. |
| `microbenchmark_confirmed` | bool | `true` only for the one token also present in `022`'s `round3_results.json` (a 3-run confirmed measurement), `false` for the rest (single Round-2 measurement only) — carried through for context, not a shortlist filter. |
| `correctness_all_shapes_pass` | bool | Must be `true` (from `022`'s own gate) to be eligible for this feature's shortlist at all (spec FR-004). |
| `shape_family` | enum | `"8B"` for every candidate in this feature's shortlist (research.md Decision 2 — `022`'s scores are all keyed to the 8B-shaped representative shapes). |
| `model_used` | string | Resolved PTE filename for e2e measurement, derived from `shape_family` — `llama3_1_8b_4w_buffer_ctx3072.pte` for every shortlist candidate. |
| `shortlisted` | bool | `true` for all 8 of `022`'s Round-2 correctness-passing candidates (research.md Decision 1 — no further trimming needed since the source pool is already ≤8). |

`BASELINE_TOKEN = "(unset — default dispatch)"`: the currently-shipped fixed 4w dispatch
is represented as a `Candidate` with no `ET_VK_Q4GSW_COOPMAT_VARIANT` value set, so it can
be compared using the same `E2EMeasurement` shape as every swept candidate.

## E2EMeasurement

One real end-to-end run for a `Candidate` on a specific model PTE, with the full existing
`dev` optimization stack (WMMA coopmat linear/SDPA, node-threshold workaround) enabled.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `Candidate.token` (or `BASELINE_TOKEN`). |
| `model_stage` | enum | `"8b_search"` (User Stories 1-3) \| `"1b3b_confirmation"` (User Story 4) — distinguishes the primary search from the post-hoc cross-size confirmation pass (research.md Decision 2). |
| `stage` | enum | `"screen"` (1 run) \| `"confirm"` (3 fresh runs, research.md Decision 3). |
| `run_index` | int | 1-indexed within its `stage` (always `1` for `screen`; `1`-`3` for `confirm`). |
| `prefill_tok_s` | float | Read directly from the runner's `PyTorchObserver` JSON line — never estimated (Principle VI). |
| `decode_tok_s` | float \| null | Recorded opportunistically (spec Assumptions: prefill is the primary metric). |
| `model_used` | string | For `model_stage: "8b_search"`, always `llama3_1_8b_4w_buffer_ctx3072.pte`; for `model_stage: "1b3b_confirmation"`, `llama3_2_1b_4w_buffer_ctx3072.pte` or `llama3_2_3b_4w_buffer_ctx3072.pte`. |
| `driver_hash` | string | Verified driver identity at measurement time (Principle VIII). |
| `board` | string | Which M5 EVT1 board produced this result. |
| `clocks_pinned` | bool | Whether the pin was verified bound (Principle VII). |
| `coherence_checked` | bool | Whether a short-prompt sanity check was run for this `candidate_token`+`model_used` combination before trusting its timing numbers — required once per combination. |

**Derived**: `screen_ratio(candidate, model_stage)` = `(mean(screen prefill_tok_s) -
mean(baseline screen prefill_tok_s for the same model_stage)) / mean(baseline screen
prefill_tok_s for the same model_stage)`. `escalate_to_confirm(candidate, model_stage)` =
`screen_ratio(candidate, model_stage) >= -0.10` (research.md Decision 3), applied
independently per `model_stage` — an 8B escalation decision does not carry over to the
1B/3B pass automatically.

## ConfirmationResult

The 3-run statistical summary for a candidate that screened within 10% of, or ahead of,
baseline, within a given `model_stage`.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | FK to `Candidate.token`. |
| `model_stage` | enum | `"8b_search"` \| `"1b3b_confirmation"`. |
| `model_used` | string | The specific PTE this confirmation was run against. |
| `mean_prefill_tok_s` | float | Mean of the 3 `confirm`-stage `E2EMeasurement.prefill_tok_s` values for this `(candidate_token, model_stage, model_used)`. |
| `stddev_prefill_tok_s` | float | Stddev of the same 3 values. |
| `cov` | float | `stddev / mean`. |
| `baseline_mean_prefill_tok_s` | float | The shipped baseline's own confirmed 3-run mean for the same `model_used` (baseline is always confirmed, regardless of screen-ratio threshold — spec FR-011: the comparison target itself must be real, multi-run evidence). |
| `improvement_pct` | float | `(mean_prefill_tok_s - baseline_mean_prefill_tok_s) / baseline_mean_prefill_tok_s * 100`. |

## RankAgreementFinding

The explicit statement of whether `022`'s microbenchmark rank predicts the new 8B e2e
rank (spec SC-006) — independent of which candidate wins.

| Field | Type | Notes |
|---|---|---|
| `agreement` | enum | `"agree"` \| `"partially_agree"` \| `"disagree"`. |
| `microbenchmark_top1` | string | `Candidate.token` ranked #1 by `microbenchmark_gflops`. |
| `e2e_top1` | string | `Candidate.token` (or `BASELINE_TOKEN`) ranked #1 by confirmed 8B e2e `prefill_tok_s` (or, if none was confirmed, the best screening result). |
| `rationale` | string | One-paragraph explanation citing the specific rank positions that agree/disagree. |

## CrossSizeFinding (new — User Story 4 / spec SC-007)

The explicit per-model-size statement of whether the final 8B answer's config holds on
1B and 3B.

| Field | Type | Notes |
|---|---|---|
| `final_8b_config` | string | The `Candidate.token` (or `BASELINE_TOKEN`) reported as the definitive 8B answer (User Story 3). |
| `model_size` | enum | `"1B"` \| `"3B"`. |
| `direction` | enum | `"holds"` (still a win/loss in the same direction as the 8B finding, within the same statistical bar) \| `"neutral"` (within noise of baseline on this size) \| `"reverses"` (opposite direction from the 8B finding). |
| `improvement_pct` | float | This model size's own improvement (or regression) over its own baseline, using the same `ConfirmationResult`-derived formula. |

## SearchExtension (only if User Story 2 triggers)

Records a new, previously-unmeasured candidate added because the initial 8-candidate
8B shortlist failed to beat baseline.

| Field | Type | Notes |
|---|---|---|
| `candidate_token` | string | New token, following the same `tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` naming convention. |
| `selection_rationale` | string | Documented reason this specific new candidate was chosen (spec FR-007), derived from `022`'s analytical scoring model plus the `RankAgreementFinding`. |
| `budget_consumed` | int | Running count of extension candidates measured so far, checked against the pre-declared additional budget (spec FR-007/FR-009). |
