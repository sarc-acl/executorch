# Contract: E2E Ranking Pipeline File Interfaces

This feature is a small chain of scripts communicating through files, driving real `adb`
measurement rounds — not a network or library API. This document is the contract between
them (and for the human reading the final report). It extends `specs/027`'s established
schema shape with two new stages this feature adds: `port_verification.json` (Decision 0)
and the 1B/3B `model_stage` dimension threaded through every downstream file (Decision 2).

## -1. `port_verification.json` (output of the one-time infra port + correctness re-check — new, `027` had no equivalent)

A JSON array of `PortVerification` records (data-model.md), one per shortlisted
candidate token, produced by re-running `022`'s existing correctness harness against the
newly-ported `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` on the `028-4w-e2e-tile-sweep`
execution branch.

```json
[
  {
    "port_commit": "<sha of the port commit>",
    "base_shader_reference": "linear_q4gsw_coop.glsl",
    "archived_patch_reference": ".archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.glsl",
    "candidate_token": "tsweep_t128x64k16g14s32",
    "correctness_status": "pass"
  }
]
```

**Contract**: this file MUST exist and every shortlisted token MUST have
`correctness_status: "pass"` here before `build_prefilter_ranking.py`'s output may be
consumed by any screening script — a candidate with `correctness_status: "fail"` or with
no entry at all here MUST NOT receive any `E2EMeasurement` (spec FR-004 as extended by
plan.md's Testing section).

## 0. `prefilter_ranking.json` (output of `build_prefilter_ranking.py`, consumes `022`'s results directly)

A JSON array of `Candidate` records (data-model.md), one per correctness-verified entry
from `022`'s `round2_results.json` (cross-referenced against `round3_results.json` for
the `microbenchmark_confirmed` flag), sorted by `microbenchmark_rank` ascending.

```json
[
  {
    "token": "tsweep_t128x64k16g14s32", "source_feature": "022",
    "microbenchmark_gflops": 2518.77, "microbenchmark_rank": 1,
    "microbenchmark_confirmed": true, "correctness_all_shapes_pass": true,
    "shape_family": "8B", "model_used": "llama3_1_8b_4w_buffer_ctx3072.pte",
    "shortlisted": true
  },
  {
    "token": "tsweep_t64x128k16g41s32", "source_feature": "022",
    "microbenchmark_gflops": 2362.0, "microbenchmark_rank": 2,
    "microbenchmark_confirmed": false, "correctness_all_shapes_pass": true,
    "shape_family": "8B", "model_used": "llama3_1_8b_4w_buffer_ctx3072.pte",
    "shortlisted": true
  }
]
```

**Contract**: every entry has `correctness_all_shapes_pass: true` — a candidate that
failed correctness in `022` MUST NOT appear here at all. All 8 entries have
`shortlisted: true` (research.md Decision 1 — `022`'s Round 2 pool is already exactly 8,
so no trimming logic runs; if a future re-derivation of this file finds more than 8
correctness-passing candidates, this contract requires trimming to the top 8 by
`microbenchmark_rank` and stating that trim explicitly).

## 1. `screen_results.json` (output of `run_e2e_screen.py`, consumes `prefilter_ranking.json` + `port_verification.json`)

One `E2EMeasurement` (`stage: "screen"`, `model_stage: "8b_search"`) per shortlisted
candidate whose `port_verification.json` entry is `correctness_status: "pass"`, PLUS one
for `BASELINE_TOKEN`.

```json
[
  {
    "candidate_token": "(unset — default dispatch)", "model_stage": "8b_search",
    "stage": "screen", "run_index": 1, "prefill_tok_s": 131.24, "decode_tok_s": null,
    "model_used": "llama3_1_8b_4w_buffer_ctx3072.pte",
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80", "board": "xgpusw-debug08",
    "clocks_pinned": true, "coherence_checked": true
  },
  {
    "candidate_token": "tsweep_t128x64k16g14s32", "model_stage": "8b_search",
    "stage": "screen", "run_index": 1, "prefill_tok_s": 128.90, "decode_tok_s": null,
    "model_used": "llama3_1_8b_4w_buffer_ctx3072.pte",
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80", "board": "xgpusw-debug08",
    "clocks_pinned": true, "coherence_checked": true
  }
]
```

**Contract**: `model_used` on every entry MUST equal the corresponding
`Candidate.model_used` from `prefilter_ranking.json`. `coherence_checked: true` MUST be
set at least once per distinct `model_used` value before any `prefill_tok_s` under that
model is trusted (Principle VI). A candidate absent from `port_verification.json` or with
`correctness_status: "fail"` there MUST NOT appear in this file.

## 2. `escalation_decisions.json` (derived from `screen_results.json`, precedes confirmation)

A JSON array recording, for every shortlisted candidate, whether it was escalated to
confirmation and why — scoped to `model_stage: "8b_search"` only at this point.

```json
[
  {"candidate_token": "tsweep_t128x64k16g14s32", "model_stage": "8b_search", "screen_ratio": -0.018, "escalated": true},
  {"candidate_token": "tsweep_t64x128k16g41s32", "model_stage": "8b_search", "screen_ratio": -0.22, "escalated": false}
]
```

**Contract**: `escalated = (screen_ratio >= -0.10)` (research.md Decision 3),
deterministic from `screen_results.json` — no manual override.

## 3. `confirmation_results.json` (output of `run_e2e_confirm.py`, consumes `escalation_decisions.json`)

One `ConfirmationResult` per escalated candidate PLUS the baseline (always confirmed —
spec FR-011).

```json
[
  {
    "candidate_token": "tsweep_t128x64k16g14s32", "model_stage": "8b_search",
    "model_used": "llama3_1_8b_4w_buffer_ctx3072.pte",
    "mean_prefill_tok_s": 129.4, "stddev_prefill_tok_s": 0.6, "cov": 0.0046,
    "baseline_mean_prefill_tok_s": 131.1, "improvement_pct": -1.3
  }
]
```

**Contract**: `baseline_mean_prefill_tok_s` MUST come from a real 3-run
`ConfirmationResult` for `BASELINE_TOKEN` under the same `model_used` — never a single
screening data point (spec FR-011).

## 4. `final_8b_answer.json` (output of `build_report.py`, User Story 3)

The single, unambiguous 8B answer (spec FR-008).

```json
{
  "winner_token": "(unset — default dispatch)",
  "winner_is_baseline": true,
  "improvement_pct": 0.0,
  "rank_agreement": {
    "agreement": "disagree",
    "microbenchmark_top1": "tsweep_t128x64k16g14s32",
    "e2e_top1": "(unset — default dispatch)",
    "rationale": "..."
  },
  "candidates_measured": 8,
  "candidates_confirmed": 3,
  "excluded_candidates": [
    {"token": "tsweep_t64x64k16g41s32", "reason": "screen_ratio -0.31, below -0.10 escalation bar"}
  ]
}
```

**Contract**: `excluded_candidates` MUST cover every shortlisted candidate not present in
`confirmation_results.json`, each with a stated reason (spec SC-005) — no silent
omissions.

## 5. `cross_size_confirmation.json` (output of `run_1b3b_confirmation.py`, User Story 4 — new, `027` had no equivalent)

One `E2EMeasurement`/`ConfirmationResult` pair (as needed) plus one `CrossSizeFinding` per
model size, for `final_8b_answer.json`'s exact `winner_token`.

```json
[
  {
    "model_size": "1B", "final_8b_config": "(unset — default dispatch)",
    "direction": "holds", "improvement_pct": 0.0
  },
  {
    "model_size": "3B", "final_8b_config": "(unset — default dispatch)",
    "direction": "holds", "improvement_pct": 0.0
  }
]
```

**Contract**: exactly one `CrossSizeFinding` per model size (1B, 3B) — never omitted, even
when the 8B answer is "baseline stands" (spec FR-012, acceptance scenario 2 of User Story
4). `final_8b_config` MUST equal `final_8b_answer.json`'s `winner_token` — this file never
introduces a different candidate for 1B/3B than the one User Story 3 already settled on.

## 6. `sweep-report.md` (final human-readable report, output of `build_report.py`)

Markdown synthesis of `final_8b_answer.json` + `cross_size_confirmation.json`, following
`022`'s/`027`'s existing `sweep-report.md` convention: Environment (device/driver/clocks/
build), the 8B answer with full evidence, the rank-agreement finding (SC-006), the 1B/3B
cross-size finding (SC-007), and — if User Story 2 triggered — the extension candidates
and their outcomes.
