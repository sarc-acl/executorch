# Contract: 8da4w subgroup32-Reopen Sweep Pipeline File Interfaces

This feature is a small chain of scripts communicating through files, plus one preceding
compile-legality probe (User Story 1), not a network or library API. This document extends
`specs/025`'s contract with the two changes this feature's spec requires: `subgroup_size` is
enumerated (not fixed at 64), and correctness is a per-shape matrix (not a single verdict).
`dbuf_reconfirmation.json` from `025` is read as an input (the fixed loop structure), not
regenerated.

## 0. `subgroup32_legality.json` (output of User Story 1's on-device probe, precedes all scripts)

A JSON array of compile/pipeline-creation attempts for a representative spread of
`subgroup_size=32` candidates across multiple tile shapes (not the single shape `025`'s T014
and this session's prior probe each used).

```json
[
  {
    "candidate_token": "tsweep_t128x64k32g22s32",
    "compile_status": "compiles",
    "pipeline_creation_crashed": false,
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80",
    "board": "xgpusw-debug08"
  },
  {
    "candidate_token": "tsweep_t128x32k16g12s32",
    "compile_status": "compiles",
    "pipeline_creation_crashed": false,
    "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80",
    "board": "xgpusw-debug08"
  }
]
```

**Contract**: at least 5 distinct tile shapes are attempted (a materially broader spread than
either prior single-shape probe — spec User Story 1 Independent Test). If any entry has
`pipeline_creation_crashed: true`, this file's own top-level summary line states this
explicitly and names which shape(s) crashed — the historical crash reproducing at *some but
not all* shapes is a valid, reportable outcome (spec Edge Cases), not treated as invalidating
the whole axis or as a script bug.

## 1. `configs.json` (output of `enumerate_configs.py`)

A JSON array of every `ConfigurationCandidate` (data-model.md) in the re-derived legal space
— `subgroup_size` now enumerated over `{32, 64}` per candidate, loop structure fixed at the
`dbuf2` winner read from `025`'s `dbuf_reconfirmation.json` — `valid=true` only.

```json
[
  {
    "token": "tsweep_t128x64k32g22s64",
    "wg_tile_m": 128, "wg_tile_n": 64, "wg_tile_k": 32,
    "sg_grid_x": 2, "sg_grid_y": 2, "subgroup_size": 64,
    "wg_size": 256, "lds_bytes": 33792, "accumulators_per_sg": 8,
    "valid": true, "compile_status": "not_attempted"
  },
  {
    "token": "tsweep_t128x64k32g22s32",
    "wg_tile_m": 128, "wg_tile_n": 64, "wg_tile_k": 32,
    "sg_grid_x": 2, "sg_grid_y": 2, "subgroup_size": 32,
    "wg_size": 128, "lds_bytes": 33792, "accumulators_per_sg": 8,
    "valid": true, "compile_status": "not_attempted"
  }
]
```

**Contract**: entries with `subgroup_size: 32` **are now legal to appear** (the opposite
contract from `025`'s file, which forbade them — research.md Decision 1). Every
`(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y)` combination that was legal in `025`'s
space appears twice here (once per `subgroup_size` value) unless a subgroup-size-specific
constraint (e.g. `wg_size <= 1024`) rules one out — that exclusion is recorded with its own
reason, not silently omitted. Total entry count is `total_valid_universe`
(data-model.md `SearchBudget`) — expected roughly double `025`'s 542 before any
compile-status pruning, an actual script output not assumed.

## 2. `shortlist.json` (output of `score_and_shortlist.py`, consumes `configs.json`)

A JSON array of `AnalyticalScore` records, one per candidate in `configs.json`, sorted by
`rank` ascending — unchanged structure from `025`, now ranking across both subgroup sizes
together (the analytical score already accounts for `wg_size`, which is subgroup-size
sensitive, per data-model.md).

```json
[
  {
    "candidate_token": "tsweep_t128x32k16g12s64",
    "occupancy_proxy": 2.10, "register_penalty": 1.0, "score": 2.10,
    "rank": 1, "shortlisted": true, "shortlist_reason": "anchor:025-winner"
  }
]
```

**Contract**: every candidate from `configs.json` appears exactly once. `025`'s standing
winner (`tsweep_t128x32k16g12s64`, `subgroup_size=64`) is always `shortlisted: true`
regardless of rank (`shortlist_reason: "anchor:025-winner"`), replacing `025`'s own
`anchor:shipped-config`/`anchor:4w-winner` anchors (both still included too, for continuity —
this file's anchor set is additive, not a replacement of `025`'s). `shortlisted: true` count
never exceeds `budget.json`'s `budget_cap` plus the number of anchors.

## 3. `correctness_matrix.json` (output of the broadened correctness stage, consumes `shortlist.json`)

**New file in this feature** — has no `025` equivalent, since `025` folded a single-shape
correctness check into each `MeasurementResult` directly. A JSON array of `CorrectnessResult`
records (data-model.md), one per shortlisted candidate that compiles.

```json
[
  {
    "candidate_token": "tsweep_t128x64k32g22s32",
    "per_shape_results": {
      "M128_K128_N128": "pass",
      "M256_K256_N256": "fail",
      "M256_K128_N128": "fail",
      "M256_K128_N64": "fail",
      "M128_K4096_N128": "pass"
    },
    "all_shapes_pass": false,
    "failing_shapes": ["M256_K256_N256", "M256_K128_N128", "M256_K128_N64"],
    "dispatch_confirmed": true
  },
  {
    "candidate_token": "tsweep_t128x32k16g12s64",
    "per_shape_results": {
      "M128_K128_N128": "pass",
      "M256_K256_N256": "pass",
      "M256_K128_N128": "pass",
      "M256_K128_N64": "pass",
      "M128_K4096_N128": "pass"
    },
    "all_shapes_pass": true,
    "failing_shapes": [],
    "dispatch_confirmed": true
  }
]
```

**Contract**: every candidate's `per_shape_results` map has an identical key set (the full
representative shape set, research.md Decision 2) — a candidate is never reported with fewer
shapes tested than another, which would silently reintroduce the single-shape gap this
feature exists to close. Only candidates with `all_shapes_pass: true` may appear in
`round{1,2,3}_results.json` below (data-model.md's ranking-eligibility rule) — the staged
search script MUST refuse to emit a `MeasurementResult` for a candidate whose
`CorrectnessResult.all_shapes_pass` is `false` or missing.

## 4. `round{1,2,3}_results.json` (output of `staged_search.py`, consumes `correctness_matrix.json`)

Same shape as `025`'s equivalent file, with the added `correctness_ref`/`board` fields from
data-model.md's extended `MeasurementResult`. One file per round; `round2`/`round3` only
contain candidates that survived the prior round.

**Contract**: every entry's `correctness_ref` resolves to a `CorrectnessResult` in
`correctness_matrix.json` with `all_shapes_pass: true` — an entry whose `correctness_ref`
fails this check is a contract violation (a bug in the orchestration script, not a valid
result) and must not appear.

**Halt contract**: identical to `025` — if a round detects a driver-hash mismatch or device
unavailability mid-round, the script writes out whatever records were already collected plus
a top-level `{"halted": true, "halt_reason": "..."}` sentinel, never a silently-truncated
file. If both M5 EVT1 boards are used across rounds, each round's records carry their own
`board` value (data-model.md) — a halt on one board does not implicitly apply to the other.

## 5. `budget.json` (output of `staged_search.py`, updated after each round)

A single `SearchBudget` record (data-model.md), rewritten after every round.

**Contract**: `configs_measured_on_hardware` MUST never exceed `budget_cap`
(`min(round(0.15 * total_valid_universe), 30)`, unchanged convention — research.md Decision
3); the script MUST refuse to start a new round if doing so would exceed `budget_cap`.
`LoopStructureResult` measurements (not re-run, read from `025`), the User Story 1 legality
probe (`subgroup32_legality.json`), and the `correctness_matrix.json` stage are all excluded
from this count — only performance-measurement rounds (`round{1,2,3}`) count against the cap,
per data-model.md's `SearchBudget.configs_measured_on_hardware` note.

## 6. `sweep-report.md` (final output, human-facing)

A Markdown report, extending `025`'s equivalent structure:

- A one-paragraph summary stating the `axis_disposition` (data-model.md
  `OptimalConfiguration`) — whether a subgroup=32 candidate wins, is legal-but-not-improving,
  or is confirmed illegal — before any other detail (spec User Story 3 Acceptance Scenario 3).
- A "subgroup=32 legality" section: the `subgroup32_legality.json` results, naming any shape
  where the historical crash did or didn't reproduce.
- A "correctness matrix" section: one row per candidate that compiled, one column per
  representative shape, `pass`/`fail` per cell — the artifact that makes shape-dependent
  correctness visible at a glance instead of buried in a single overall verdict (the specific
  gap in `025`'s T014 and this session's own prior probe).
- A "speedup vs `025`'s winner" table, one row per Round-3 finalist (both subgroup sizes, if
  both reach Round 3), per representative shape and overall — the primary comparison this
  feature adds (spec SC-003).
- Carried-forward comparisons against the pre-`025` shipped baseline and `4w`'s winner, for
  continuity with `025`'s report.
- A "search cost" section: `configs_measured_on_hardware`, `budget_cap`,
  `total_device_seconds`.
- A "probe disposition" section stating explicitly whether the session's ad-hoc `sg32test`
  shader/binding was superseded-and-removed or retained with reason (spec FR-012/SC-007).
- A "shader comment update" section: the proposed diff to
  `linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s header comment (research.md Decision 6), or a
  statement of why it isn't included if genuinely deferred.
- A pruning-audit appendix (or link to `shortlist.json`/`correctness_matrix.json`) so any
  candidate's fate can be traced without re-running the search.

**Contract**: this file is the only artifact a reader needs to open to get the feature's
answer — everything else is supporting/audit data.
