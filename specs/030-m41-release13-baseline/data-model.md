# Data Model: M41 Release/1.3 Baseline Clock & Quant-Mode Study

## Run

One execution of `llama_main_rel1.3` against a single (model, quant mode, clock mode) combination,
at one rep index. 36 total across this feature (9 already collected + 27 new, per spec SC-007).

| Field | Type | Notes |
|---|---|---|
| `model` | enum | `llama3_2_1b` / `llama3_2_3b` / `llama3_1_8b` |
| `quant_mode` | enum | `4w` / `8da4w` |
| `clock_mode` | enum | `pinned` (509/2730/663 MHz) / `floating` (DVFS unpinned, full hardware range) |
| `rep_index` | int | 1–3 |
| `outcome` | enum | `ok` / `crashed` / `dvfs_artifact` — the third value is set when a `clock_mode=pinned` run's sysfs write succeeded but FR-009's throughput cross-check failed (prefill_tok_s exceeded 70% of the floating number for the same cell); it is a distinct, non-numeric, non-"crashed" outcome, per spec FR-012/Edge Cases |
| `prefill_tok_s` | float\|null | populated even when `outcome=dvfs_artifact` (the measured — but disqualified — throughput, shown in the report per FR-012); null only if `outcome=crashed` |
| `decode_tok_s` | float\|null | same rule as `prefill_tok_s` |
| `crash_signature` | string\|null | e.g. `VK_ERROR_DEVICE_LOST (vkQueueWaitIdle=-4)`; null unless `outcome=crashed` |
| `crash_cause` | enum\|null | `gpu_watchdog` / `host_oom` / `unknown` — set per Decision 4's `dmesg`/`meminfo` check, never assumed; null unless `outcome=crashed` |
| `pin_verified` | bool\|null | For `clock_mode=pinned` only: true only if both the sysfs readback AND the throughput cross-check (spec FR-009) confirm it — false when `outcome=dvfs_artifact`; null for floating runs |

## ModelSummary

One (model, quant_mode, clock_mode) combination — the per-model aggregate row each results table
ultimately reports alongside its 3 individual rep "cells" (spec FR-012's terminology — note this
entity is named `ModelSummary`, not "Cell", specifically to avoid colliding with that per-rep
usage). Derived from its constituent Runs, never entered directly.

| Field | Type | Notes |
|---|---|---|
| `model` | enum | as above |
| `quant_mode` | enum | as above |
| `clock_mode` | enum | as above |
| `runs` | Run[3] | the 3 reps summarized here |
| `n_valid` | int | count of `runs` with `outcome=ok` (a `dvfs_artifact` run is NOT valid for mean/CoV purposes, same as `crashed`) |
| `prefill_mean` | float\|null | mean of valid runs' `prefill_tok_s`; null if `n_valid=0` |
| `prefill_cov` | float\|null | stdev/mean × 100%; null if `n_valid<2` (spec FR-011) |
| `decode_mean` | float\|null | mean of valid runs' `decode_tok_s`; null if `n_valid=0` |
| `decode_cov` | float\|null | stdev/mean × 100%; null if `n_valid<2` |
| `mean_caveat` | string\|null | required non-null text for `clock_mode=floating` summaries per spec FR-007 ("mean may mix cold-start peak with throttled steady state"); null for pinned summaries |

## Table

One of the four deliverables (spec FR-007/FR-012): `4w-pinned`, `4w-floating`, `8da4w-pinned`,
`8da4w-floating`. Each Table has exactly 3 ModelSummaries (one per model), each showing its 3 Runs
individually as one of 9 rep-cells (a number, "CRASHED", or "DVFS-ARTIFACT" per rep, per FR-012)
plus the derived `prefill_mean`/`prefill_cov`/`decode_mean`/`decode_cov`.

## Known values as of plan time (pre-existing, from this session's earlier work)

The `4w-floating` table's 9 Runs are already fully determined (not re-run by this feature — spec
FR-002):

| model | rep | outcome | prefill_tok_s | decode_tok_s |
|---|---|---|---:|---:|
| llama3_2_1b | 1 | ok | 594.14 | 30.14 |
| llama3_2_1b | 2 | ok | 601.82 | 30.41 |
| llama3_2_1b | 3 | ok | 601.47 | 30.55 |
| llama3_2_3b | 1 | ok | 219.46 | 13.02 |
| llama3_2_3b | 2 | ok | 215.24 | 13.00 |
| llama3_2_3b | 3 | ok | 212.18 | 13.01 |
| llama3_1_8b | 1 | ok | 90.91 | 7.26 |
| llama3_1_8b | 2 | **crashed** | — | — |
| llama3_1_8b | 3 | ok | 86.90 | 7.21 |

`llama3_1_8b` rep 2's `crash_cause` is `unknown` (attributed to thermal by inference at collection
time, per this session's own discussion). The Decision 4 retroactive check was attempted during
implementation but was **inconclusive** — the on-device `dmesg` ring buffer only retains ~210s of
uptime history, well short of reaching back to this crash. A live reproduction of the identical
error signature (T004's probe) confirmed `gpu_watchdog`, not `host_oom`, as the mechanism for that
occurrence — see `results/m41-release13-baseline-report.md`'s Methodology notes — but that is
evidence about the mechanism in general, not a settled attribution for this specific historical
rep.

The remaining 27 Runs (4w-pinned, 8da4w-floating, 8da4w-pinned — 3 models × 3 reps each) are all
`not_yet_run` as of plan time; this feature's implementation (`/speckit-tasks` → execution) fills
them in.

## Lifecycle

```
Run created (not_yet_run)
  --(execute per quickstart.md, apply Decision 3/4 as needed)-->
  outcome = ok (prefill/decode populated)
          | crashed (crash_signature/crash_cause populated)
          | dvfs_artifact (prefill/decode populated but pin_verified=false, per FR-009's threshold)
  --(3 Runs per ModelSummary complete)-->
ModelSummary's prefill_mean/cov, decode_mean/cov computed from valid (outcome=ok) Runs only
  (n_valid>=1 for mean, >=2 for CoV)
  --(all 3 ModelSummaries per Table complete)-->
Table assembled into results/m41-release13-baseline-report.md
  --(all 4 Tables complete)-->
done (spec SC-007)
```
