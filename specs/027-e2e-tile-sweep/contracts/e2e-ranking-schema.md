# Contract: E2E Ranking Pipeline File Interfaces

This feature is a small chain of scripts communicating through files, driving real `adb`
measurement rounds — not a network or library API. This document is the contract between
them (and for the human reading the final report).

## 0. `prefilter_ranking.json` (output of `build_prefilter_ranking.py`, consumes `025`/`026` results directly)

A JSON array of `Candidate` records (data-model.md), one per correctness-verified entry
from `025`'s `round2_results.json`/`round3_results.json` and `026`'s
`correctness_matrix.json`/`round3_results.json`, sorted by `microbenchmark_rank` ascending.

```json
[
  {
    "token": "tsweep_t64x64k16g21s32", "source_feature": "026", "subgroup_size": 32,
    "microbenchmark_gflops": 2207.2, "microbenchmark_rank": 1,
    "correctness_all_shapes_pass": true, "shape_family": "8B",
    "model_used": "llama3_1_8b_8da4w_buffer_ctx3072.pte", "shortlisted": true
  },
  {
    "token": "tsweep_t128x32k16g12s64", "source_feature": "025", "subgroup_size": 64,
    "microbenchmark_gflops": 1731.0, "microbenchmark_rank": 2,
    "correctness_all_shapes_pass": true, "shape_family": "8B",
    "model_used": "llama3_1_8b_8da4w_buffer_ctx3072.pte", "shortlisted": true
  }
]
```

**Contract**: every entry has `correctness_all_shapes_pass: true` — a candidate that
failed correctness at any shape in `025`/`026` MUST NOT appear here at all (not filtered
downstream, never included). Exactly 8 entries have `shortlisted: true` (spec
Clarifications) unless fewer than 8 correctness-passing candidates exist in the combined
`025`+`026` data (in which case all of them are shortlisted, and this file's own summary
line states that the shortlist is smaller than 8 and why).

## 1. `screen_results.json` (output of `run_e2e_screen.py`, consumes `prefilter_ranking.json`)

One `E2EMeasurement` (`stage: "screen"`) per shortlisted candidate, PLUS one for
`BASELINE_TOKEN` (the baseline is always screened too, as the comparison reference).

```json
[
  {
    "candidate_token": "(unset — default dispatch)", "stage": "screen", "run_index": 1,
    "prefill_tok_s": 100.73, "decode_tok_s": null,
    "model_used": "llama3_1_8b_8da4w_buffer_ctx3072.pte",
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80", "board": "xgpusw-debug08",
    "clocks_pinned": true, "coherence_checked": true
  },
  {
    "candidate_token": "tsweep_t64x64k16g21s32", "stage": "screen", "run_index": 1,
    "prefill_tok_s": 98.29, "decode_tok_s": null,
    "model_used": "llama3_1_8b_8da4w_buffer_ctx3072.pte",
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80", "board": "xgpusw-debug08",
    "clocks_pinned": true, "coherence_checked": true
  }
]
```

**Contract**: `model_used` on every entry MUST equal the corresponding `Candidate.model_used`
from `prefilter_ranking.json` (research.md Decision 2's enforcement point — a mismatch here
is the specific bug this feature exists partly to prevent, and the script MUST refuse to
proceed if it detects one rather than silently recording a shape-mismatched number).
`coherence_checked: true` MUST be set at least once per distinct `model_used` value before
any `prefill_tok_s` under that model is trusted (Principle VI).

## 2. `escalation_decisions.json` (derived from `screen_results.json`, precedes confirmation)

A JSON array recording, for every shortlisted candidate, whether it was escalated to
confirmation and why.

```json
[
  {"candidate_token": "tsweep_t64x64k16g21s32", "screen_ratio": -0.027, "escalated": true},
  {"candidate_token": "tsweep_t128x64k16g41s32", "screen_ratio": -0.31, "escalated": false}
]
```

**Contract**: `escalated = (screen_ratio >= -0.10)` (research.md Decision 3) — deterministic
from `screen_ratio`, never manually overridden. Every shortlisted candidate appears exactly
once, satisfying spec SC-005 for the screening stage.

## 3. `confirm_results.json` (output of `run_e2e_confirm.py`, consumes `escalation_decisions.json`)

One `E2EMeasurement` (`stage: "confirm"`, `run_index` 1-3) per escalated candidate, PLUS
3 for `BASELINE_TOKEN` (baseline is always confirmed — data-model.md `ConfirmationResult`
note).

**Contract**: contains measurements ONLY for candidates with `escalated: true` in
`escalation_decisions.json`, plus the baseline — a non-escalated candidate MUST NOT appear
here (this is the device-time savings the adaptive bar exists to produce).

## 4. `confirmation_results.json` (derived summary, one `ConfirmationResult` per confirmed candidate)

```json
[
  {
    "candidate_token": "tsweep_t64x64k16g21s32", "mean_prefill_tok_s": 98.10,
    "stddev_prefill_tok_s": 0.16, "cov": 0.0016,
    "baseline_mean_prefill_tok_s": 100.65, "improvement_pct": -2.53,
    "beats_baseline": false
  }
]
```

**Contract**: `beats_baseline` is computed exactly per data-model.md's derived rule
(positive `improvement_pct` AND non-overlapping mean±stddev ranges) — never asserted by
hand.

## 5. `extension_candidates.json` (User Story 2, only present if triggered)

Present only if every entry in `confirmation_results.json` has `beats_baseline: false`.
Contains `SearchExtensionBudget` (data-model.md) plus one `Candidate`-shaped entry per
newly-selected configuration, each carrying `shortlisted: true` and a `selection_rationale`
inherited from the budget record.

**Contract**: if this file exists, `new_candidates_selected <= budget_cap`, and every entry
in it flows through the same `screen_results.json` → `escalation_decisions.json` →
`confirm_results.json` pipeline as the original 8 — no shortcut path for extension
candidates.

## 6. `sweep-report.md` (final output, human-facing)

A Markdown report:

- A one-paragraph `FinalAnswer` summary (data-model.md) stated first: the winning token
  (which may be `BASELINE_TOKEN`) and, if not baseline, its confirmed improvement
  percentage.
- The `microbenchmark_vs_e2e_rank_agreement` finding (spec SC-006), stated as its own
  paragraph — this is new information independent of which token wins.
- A screening-stage table: every shortlisted candidate, its screen ratio, and whether it
  escalated.
- A confirmation-stage table: every escalated candidate's 3-run mean/stddev/CoV and
  `beats_baseline` verdict, alongside the baseline's own confirmed numbers.
- If User Story 2 triggered: a section covering the extension candidates using the same
  two tables.
- A "search cost" section: total distinct candidates measured (screen + confirm, User
  Story 1 + any extension), reported against the target "far fewer than the full legal
  space" bar (spec SC-004).
- A skip-reasons appendix (or link to `prefilter_ranking.json`) so any correctness-passing
  `025`/`026` candidate NOT in the top-8 shortlist can be traced (spec SC-005).

**Contract**: this file is the only artifact a reader needs to open to get the feature's
answer — everything else is supporting/audit data.
