# Data Model: 8da4w Coopmat Tile/Subgroup Parameter Sweep

## Swept Configuration

One of the 13 entries in `research.md` Decision 4's curated set: 11 new
performance candidates + 1 reused shipped baseline + 1 deliberate
negative-test configuration (added during `/speckit-analyze` remediation,
finding G1). Of the 11 performance candidates, 6 (subgroup 32: configs 1,
3, 5, 7, 9, 11) were dropped mid-implementation after a real correctness
bug (research.md Decision 4's implementation revision) -- **5 remain
active candidates** (2, 4, 6, 8, 10), all subgroup 64.

| Field | Type | Notes |
|---|---|---|
| `config_id` | int | 0 (shipped baseline, reused from `007`) through 12 (the deliberate negative test) |
| `wg_tile_m` / `wg_tile_n` / `wg_tile_k` | int | Workgroup tile shape |
| `subgroup_size` | int | 32 or 64 |
| `sg_grid_x` / `sg_grid_y` | int | Always 2/2 for this curated set (Decision 4) |
| `kernel_name` | string | The compiled variant's name (production name for config 0, this feature's own test-owned name for configs 1-12) |
| `role` | enum | `baseline` (config 0, reused not rebuilt) / `candidate` (configs 2, 4, 6, 8, 10 -- ranked for the recommendation) / `excluded_buggy` (configs 1, 3, 5, 7, 9, 11 -- subgroup-32 correctness bug, built but not ranked or reported as timings) / `negative_test` (config 12, expected to fail correctness -- proves the check works, per research.md Decision 4's revision) |

There is no separate design-time "invalid" state: `config_id=12` is a
real, buildable, dispatchable configuration whose *expected* Sweep-Phase
Result outcome is `correctness_failure` (research.md Decision 4) -- it is
not filtered out before being attempted, only excluded from the
performance ranking regardless of its outcome. Configs 1/3/5/7/9/11 are a
different case: their `correctness_failure` was NOT expected in advance
(unlike config 12) -- discovered during T010, and excluded from ranking
for that reason, reported explicitly as an unresolved finding rather than
silently dropped (research.md Decision 4).

## Sweep-Phase Result (US2)

One entry per (active candidate, representative shape), plus
`config_id=12` (the negative test) measured at exactly one shape since
there is nothing further to learn from repeating a known-broken kernel:
5 active candidates x 6 shapes (research.md Decision 3) + 1 negative test
x 1 shape = **31 total for the sweep phase**. Configs 1/3/5/7/9/11's
partial data (config 1's own 6-shape run, which surfaced the subgroup-32
bug) is retained and reported separately as a called-out finding, not
folded into this count or the ranking.

| Field | Type | Notes |
|---|---|---|
| `config_id` | int | FK to Swept Configuration |
| `model` | string | Which model this shape belongs to |
| `op` | string | `wq` (square) or `w1_gate` (rectangular) -- the 2 representative ops per model (research.md Decision 3); always `wq` for `config_id=12`'s single shape |
| `m` / `k` / `n` | int | Shape |
| `outcome` | enum | `measured` / `compile_failure` / `pipeline_crash` / `correctness_failure` -- no separate "invalid" value; a mathematically incompatible combination (like `config_id=12`) surfaces as `correctness_failure`, not a distinct category (research.md Decision 4's revision) |
| `mean_us` / `stdev_us` / `iterations` | float/float/int | Present only if `outcome == measured` |
| `dispatch_confirmed` | bool | Kernel name contains the expected coopmat family marker (FR-005) |
| `correctness_verified` | bool | Exact int32-accumulation reference comparison passed (research.md Decision 5); expected `false` for `config_id=12` -- if it comes back `true` instead, that is itself a critical finding about the correctness check's reliability, not a passing result |
| `failure_detail` | string or null | The actual compiler/driver/correctness error, present whenever `outcome != measured` (FR-004) |

## Full-Catalog Validation Result (US3)

Only for the winning configuration(s) identified from the Sweep-Phase
Results among the 5 active candidates (`config_id` 2, 4, 6, 8, 10 --
never 0, 12, or the excluded subgroup-32 configs) -- one entry per
(winning config, model, op) across all 3 models x 7 `8da4w` ops (21 cases,
matching `007`'s full catalog).

| Field | Type | Notes |
|---|---|---|
| `config_id` | int | The winning Swept Configuration |
| `model` / `op` | string | One of `007`'s 21 `8da4w` cases |
| `mean_us` / `stdev_us` / `iterations` | float/float/int | |
| `speedup_vs_shipped_pct` | float | Against `007`'s measured shipped-config number for this exact (model, op) |
| `speedup_vs_tiled_pct` | float | Against `004`'s measured Buffer-storage tiled number for this exact (model, op) |
| `significance` | enum | `real_effect` / `noise`, via the established non-overlapping `mean +/- 2*stdev` band rule |

## Optimal Configuration Recommendation

The single top-level conclusion (US3): either one Swept Configuration
(identified by `config_id`, its tile/subgroup parameters, and its
full-catalog validation numbers) recommended as the best found, or an
explicit "no configuration in the sweep outperforms the tiled baseline"
finding (FR-007) if none of the Full-Catalog Validation Results show a
positive `speedup_vs_tiled_pct` with `real_effect` significance.

No lifecycle/state transitions -- this is a one-shot sweep-and-report,
same shape as every prior microbenchmark feature in this workstream.
