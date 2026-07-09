# Contract: 8da4w Sweep Pipeline File Interfaces

This feature is a small chain of scripts communicating through files, plus one preceding
on-device measurement round (User Story 1), not a network or library API. This document is
the contract between them (and for the human reading the final report), so each stage can
be re-run independently against a previous stage's output without re-deriving its format.

## 0. `dbuf_reconfirmation.json` (output of User Story 1's on-device round, precedes all scripts)

A JSON array of exactly four `LoopStructureResult` records (data-model.md), one per
`dbuf1`-`dbuf4`, measured at the currently-shipped 128×64/K32/2×2/s64 geometry.

```json
[
  {
    "variant": "dbuf1", "dispatch_confirmed": true, "correctness_status": "pass",
    "mean_us": 812.4, "cov": 0.021, "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80",
    "clocks_pinned": true, "failure_reason": null
  },
  {
    "variant": "dbuf2", "dispatch_confirmed": true, "correctness_status": "pass",
    "mean_us": 798.1, "cov": 0.018, "driver_hash": "c9861e9906d03fa2c7d48b804e1a1c80",
    "clocks_pinned": true, "failure_reason": null
  }
]
```

**Contract**: exactly 4 entries, one per `dbuf{1..4}`, each either measured to completion
(`correctness_status` and `mean_us`/`cov` populated) or carrying a non-null `failure_reason`
— never a silently-missing entry. The fixed loop structure used by every downstream file in
this contract is `argmin(mean_us)` among entries with `correctness_status: "pass"`. This
file's own top-level summary line (written by the same step, not a separate script) states
explicitly whether that argmin equals `"dbuf2"` (spec SC-001).

## 1. `configs.json` (output of `enumerate_configs.py`)

A JSON array of every `ConfigurationCandidate` (data-model.md) in the re-derived `8da4w`
legal space — buffer-storage int8 WMMA, `subgroup_size` fixed at 64, loop structure fixed at
the winner from `dbuf_reconfirmation.json` — `valid=true` only.

```json
[
  {
    "token": "tsweep_t128x64k32g22s64",
    "wg_tile_m": 128, "wg_tile_n": 64, "wg_tile_k": 32,
    "sg_grid_x": 2, "sg_grid_y": 2, "subgroup_size": 64,
    "wg_size": 256, "lds_bytes": 33792, "accumulators_per_sg": 8,
    "valid": true, "compile_status": "not_attempted"
  }
]
```

**Contract**: every entry has `subgroup_size: 64` — an entry with `subgroup_size: 32` MUST
NOT appear (research.md Decision 1; rejected at enumeration, not filtered downstream). Total
entry count is `total_valid_universe` (data-model.md `SearchBudget`) — a script output, not
assumed equal to `022`'s 642. `token` values are globally unique and follow the
`ET_VK_DQ8CA_COOPMAT_TILE_VARIANT` naming convention
(`tsweep_t<M>x<N>k<K>g<SGX><SGY>s64`, subgroup suffix always `s64`).

## 2. `shortlist.json` (output of `score_and_shortlist.py`, consumes `configs.json`)

A JSON array of `AnalyticalScore` records, one per candidate in `configs.json`, sorted by
`rank` ascending.

```json
[
  {
    "candidate_token": "tsweep_t128x64k32g22s64",
    "occupancy_proxy": 1.94, "register_penalty": 1.0, "score": 1.94,
    "rank": 1, "shortlisted": true, "shortlist_reason": "anchor:shipped-config"
  }
]
```

**Contract**: every candidate from `configs.json` appears exactly once (full ranking, not
just the shortlist, so pruning decisions stay auditable per spec FR-009). `shortlisted: true`
count never exceeds `budget.json`'s `budget_cap` (data-model.md `SearchBudget`) plus the
number of anchors. The currently-shipped `8da4w` geometry (`anchor:shipped-config`) is always
`shortlisted: true` regardless of rank; `4w`'s 128×64/K16/2×2/s32 winner
(`anchor:4w-winner`) is included only if it is a legal `8da4w` candidate under this feature's
Validation rules — if illegal (e.g. because it implies `subgroup_size=32`), it is recorded in
a separate top-level `excluded_anchors` array with a `reason` field instead of silently
dropped (spec User Story 2, Acceptance Scenario 2).

## 3. `round{1,2,3}_results.json` (output of `staged_search.py`, consumes `shortlist.json`)

One file per round, each a JSON array of `MeasurementResult` records (data-model.md) for
every candidate still alive entering that round. Same shape and round-elimination contract
as `022`'s equivalent file: `round1_results.json` has one entry per shortlisted candidate;
`round2_results.json`/`round3_results.json` only contain entries for candidates that survived
the prior round (`eliminated_at: false`); `round3_results.json` entries MUST have non-null
`mean_gflops`, `stddev_gflops`, and `run_count == 3` with `cov < 0.05` implied by `stddev`/`mean`
(spec Clarified 2026-07-09).

**Halt contract**: identical to `022` — if a round detects a driver-hash mismatch or device
unavailability mid-round, the script writes out whatever records were already collected plus
a top-level `{"halted": true, "halt_reason": "..."}` sentinel, never a silently-truncated file.

## 4. `budget.json` (output of `staged_search.py`, updated after each round)

A single `SearchBudget` record (data-model.md), rewritten after every round.

**Contract**: `configs_measured_on_hardware` MUST never exceed `budget_cap`
(`min(round(0.15 * total_valid_universe), 30)` — spec Clarified 2026-07-09); the script MUST
refuse to start a new round (not just warn) if doing so would exceed `budget_cap` — this is
the SC-002/FR-007 enforcement point. `LoopStructureResult` measurements and the Decision 1
subgroup=32 compile-crash re-verification (research.md Decision 1, Alternatives) are excluded
from this count.

## 5. `sweep-report.md` (final output, human-facing)

A Markdown report:

- A one-paragraph summary stating the winning `dbuf` variant (with explicit agreement/
  disagreement against the user's reported `dbuf2` claim — SC-001) and the winning
  `candidate_token` (or stating explicitly that the shipped configuration stands — FR-010).
- A "loop-structure re-confirmation" table: one row per `dbuf{1..4}`, `mean_us`/`cov`/
  `correctness_status`.
- A "speedup vs shipped `8da4w`" table, one row per Round-3 finalist, per representative
  shape (`wq`+`w1_gate` × {1B,3B,8B}) and overall.
- A cross-shader comparison row/section against `4w`'s 128×64/K16/2×2/s32 winner (spec
  FR-006/SC-004).
- A correctness section stating pass/fail for the winner at the standard multi-tile
  validation shape.
- A "search cost" section reporting `configs_measured_on_hardware`, `budget_cap`,
  `total_device_seconds`, and the SC-006 comparison against `estimated_exhaustive_device_seconds`.
- A pruning-audit appendix (or link to `shortlist.json`) so any candidate's fate can be
  traced without re-running the search (FR-009, SC-005).

**Contract**: this file is the only artifact a reader needs to open to get the feature's
answer — everything else is supporting/audit data.
