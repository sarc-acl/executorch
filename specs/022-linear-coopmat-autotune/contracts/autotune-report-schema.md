# Contract: Autotune Pipeline File Interfaces

This feature is three scripts communicating through files, not a network or
library API. This document is the contract between them (and for the human
reading the final report), so each stage can be re-run independently against
a previous stage's output without re-deriving its format.

## 1. `configs.json` (output of `enumerate_configs.py`)

A JSON array of every `ConfigurationCandidate` (data-model.md) in the
buffer-storage-only, dbuf1-loop-shape universe, `valid=true` only.

```json
[
  {
    "token": "tsweep_t128x64k16g22s32",
    "wg_tile_m": 128, "wg_tile_n": 64, "wg_tile_k": 16,
    "sg_grid_x": 2, "sg_grid_y": 2, "subgroup_size": 32,
    "wg_size": 128, "lds_bytes": 16896, "accumulators_per_sg": 8,
    "valid": true, "compile_status": "not_attempted"
  }
]
```

**Contract**: exactly 642 entries, all `valid: true` (invalid combinations
are never written to this file — they're rejected during enumeration, not
filtered downstream). `token` values are globally unique and match the
`ET_VK_Q4GSW_COOPMAT_VARIANT` naming convention already in use
(`tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>`).

## 2. `shortlist.json` (output of `score_and_shortlist.py`, consumes `configs.json`)

A JSON array of `AnalyticalScore` records, one per candidate in
`configs.json`, sorted by `rank` ascending.

```json
[
  {
    "candidate_token": "tsweep_t128x64k16g22s32",
    "occupancy_proxy": 3.88, "register_penalty": 1.0, "score": 3.88,
    "rank": 1, "shortlisted": true, "shortlist_reason": "anchor:sweep-winner"
  }
]
```

**Contract**: every candidate from `configs.json` appears exactly once
(full ranking, not just the shortlist, so pruning decisions stay auditable
per spec FR-008). `shortlisted: true` count is roughly 30-40 (top ~24-32 by
analytical rank, plus every previously-measured, compiling known
configuration from `known-measurements.json` — research.md Decision 3,
revised — some of which may already be in the top ranks). Anchors
(`shortlist_reason` starting with `anchor:`) are always `shortlisted: true`
regardless of `rank`. A known configuration that previously failed to
compile is excluded with `shortlist_reason: "known_compile_failure"`
instead — never silently omitted, and never re-measured.

## 3. `round{1,2,3}_results.json` (output of `staged_search.py`, consumes `shortlist.json`)

One file per round, each a JSON array of `MeasurementResult` records for
every candidate still alive entering that round.

```json
[
  {
    "candidate_token": "tsweep_t128x64k16g22s32",
    "round": "round1_gate",
    "correctness_status": "pass",
    "shapes_measured": [[4096, 4096]],
    "gflops_per_shape": {"4096,4096": 2653.0},
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80",
    "clocks_pinned": true,
    "eliminated_at": false,
    "elimination_reason": null
  }
]
```

**Contract**: `round1_results.json` has one entry per shortlisted candidate
(every shortlisted candidate gets at least the cheap gate — spec User Story
2 Acceptance Scenario 1). `round2_results.json` and `round3_results.json`
only contain entries for candidates that survived the prior round
(`eliminated_at: false`). A candidate that appears in `round1_results.json`
with `eliminated_at: true` MUST NOT appear in `round2_results.json`, and
likewise round 2 → round 3. `round3_results.json` entries MUST have
non-null `mean_gflops`, `stddev_gflops`, and `run_count >= 3` (Constitution
Principle IV).

**Halt contract**: if a round's execution detects a driver-hash mismatch or
device unavailability mid-round (Decision 7), the script MUST still write
out whatever `MeasurementResult` records were already collected before the
halt, plus a top-level `{"halted": true, "halt_reason": "..."}` sentinel
object appended to the array — never leave a partial file with no
indication that the round was cut short.

## 4. `budget.json` (output of `staged_search.py`, updated after each round)

A single `SearchBudget` record (data-model.md), rewritten after every round.

**Contract**: `configs_measured_on_hardware` MUST never exceed 96; the
script MUST refuse to start a new round (not just warn) if doing so would
push this count over 96 — this is the SC-001 enforcement point.

## 5. `autotune-report.md` (final output, human-facing)

A Markdown report, reusing the exact table format already established in
`jira-tile-sweep.md`:

- A one-paragraph summary naming the winning `candidate_token` (or stating
  explicitly that the existing winner stands — FR-009).
- A "speedup vs dbuf1" table, one row per Round-3 finalist, matching the
  column layout: `tile config | 8B | 3B | 1B | note` (or the equivalent
  per-shape layout already used this session if per-model FFN-shape data
  isn't re-derived).
- A correctness section stating pass/fail for the winner at the standard
  multi-tile validation shape (M=K=N=256).
- A "search cost" section reporting `configs_measured_on_hardware`,
  `total_device_seconds`, and the SC-002 comparison against
  `estimated_exhaustive_device_seconds`.
- A pruning-audit appendix (or link to `shortlist.json`) so any of the 642
  candidates' fate can be traced without re-running the search (FR-008,
  SC-005).

**Contract**: this file is the only artifact a reader needs to open to get
the feature's answer — everything else is supporting/audit data.
