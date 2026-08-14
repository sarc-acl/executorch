# M51 buffer-storage coopmat A/B at max clock — 1B/3B/8B × 4w/8da4w

**Status**: report-grade (3 rounds/model, interleaved, pinned+verified clocks, driver md5 verified).
**Date**: 2026-07-23.

## Methodology

- **Device**: M51 primary board, `0000088f8e579c33` via `ssh yanwen.xu@sj1-dmckee-d01`.
- **Driver**: documented production default `f14c51b6f8`, md5 `c9861e9906d03fa2c7d48b804e1a1c80`, verified
  before the sweep. Correctness bench (`test_coopmat_linear_bench_origcm`,
  `COOPMAT_BENCH_CORRECTNESS_ONLY=1`) **16/16 PASSED** immediately before starting.
- **Clocks**: pinned at **hardware maximum** for the whole sweep — `GPU=980000 MIF=5333000 INT=934000`
  (not the usual 509/2730/663 default), verified via sysfs readback (`min==max==target` on all three
  domains). Restored to the 509/2730/663 default at the end of the sweep (verified).
- **Runner**: `llama_main_origcm` (the `dev`-branch build with the WMMA coopmat port), already staged
  on-device.
- **Storage held constant**: all 12 (model, scheme, config) combos use the **buffer**-storage
  `ctx3072` PTE — only the coopmat dispatch env var varies between arms:
  - **baseline** = buffer PTE + `ET_VK_DISABLE_COOPMAT=1` (forces the tiled/non-WMMA path)
  - **wmma** = buffer PTE, no override (coopmat/WMMA dispatches by default — the shipped/optimized path)
  - `ET_VK_EXECUTE_NODE_THRESHOLD=16` set on **every** run, both arms, all models (confirmed 8B
    watchdog-crash fix; held constant so it can't confound the baseline/wmma comparison).
- **Workload**: full 2048-token prefill + 1024-token decode (`p2048_exact.txt`, `--num_bos=1
  --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true`) — `prompt_tokens=2048` and
  `generated_tokens≈1023` verified on every one of the 36 accepted reps.
- **Interleaving**: **3 rounds per model**, each round running all 4 (scheme × config) combos once in
  the same fixed order (`4w/baseline → 4w/wmma → 8da4w/baseline → 8da4w/wmma`) before starting the
  next round — never all-of-baseline-then-all-of-wmma, per the A/B methodology's interleaving rule.
  Models run sequentially (1B → 3B → 8B), each model's PTEs staged/removed around its 3 rounds.
- **Coherence gate**: all 12 combos individually coherence-checked (`--prompt='The capital of France
  is' --seq_len=48`, expect "Paris") before any full-length rep — all 12 passed, including
  `1b/8da4w` under both configs (a prior cross-device sweep on the *vanilla* `release/1.3` runner had
  found `1B/8da4w` crashing deterministically under buffer storage; that does not reproduce on this
  `dev`-branch runner).
- **Result**: 36/36 runs completed, **0 failures, 0 crashes, 0 retries** across the full matrix.
- Script: `scripts/run_maxclock_ab.py`; raw resumable log: `maxclock_ab.jsonl`.

## Headline finding

**Coopmat/WMMA speeds up prefill substantially — 1.4×–2.7× depending on model size and quant scheme —
but has essentially no effect on decode (~1.00× throughout).** This is expected: prefill is
compute-bound (large batched matmuls benefit from the matrix-multiply unit), while decode is a
sequence of single-token matmuls that are memory-bandwidth-bound, where the coopmat dispatch path
doesn't have much to exploit. The speedup is consistently **larger for 8da4w than for 4w** at every
model size (e.g. 8B: 8da4w ×2.73 vs 4w ×1.61) — the baseline (non-WMMA) tiled path is disproportionately
slow for 8da4w specifically (this session's earlier investigation traced *why* 8da4w underperforms 4w
on the **texture** baseline to an unvalidated driver dot4/WMMA capability gap; on **buffer** storage
with WMMA enabled, that gap disappears and 8da4w's higher-arithmetic-intensity kernel pulls ahead of
4w as expected — 8da4w/wmma beats 4w/wmma at every model size below).

## Raw per-rep results

| Combo | Prefill reps (tok/s) | Prefill median | Prefill CoV | Decode reps (tok/s) | Decode median | Decode CoV |
|---|---|---|---|---|---|---|
| 1B 4w baseline | 609.89 / 610.43 / 610.80 | 610.43 | 0.07% | 26.78 / 26.79 / 26.73 | 26.78 | 0.12% |
| 1B 4w wmma | 846.63 / 844.54 / 847.68 | 846.63 | 0.19% | 26.65 / 26.73 / 26.76 | 26.73 | 0.21% |
| 1B 8da4w baseline | 428.72 / 428.99 / 428.72 | 428.72 | 0.04% | 25.66 / 25.67 / 25.59 | 25.66 | 0.17% |
| 1B 8da4w wmma | 973.38 / 970.62 / 969.70 | 970.62 | 0.20% | 25.59 / 25.59 / 25.62 | 25.59 | 0.08% |
| 3B 4w baseline | 224.41 / 225.13 / 224.73 | 224.73 | 0.16% | 13.15 / 13.08 / 13.06 | 13.08 | 0.36% |
| 3B 4w wmma | 313.25 / 313.20 / 314.06 | 313.25 | 0.15% | 13.14 / 13.09 / 13.10 | 13.10 | 0.20% |
| 3B 8da4w baseline | 154.61 / 154.47 / 154.62 | 154.61 | 0.05% | 12.68 / 12.62 / 12.63 | 12.63 | 0.26% |
| 3B 8da4w wmma | 368.35 / 366.83 / 366.30 | 366.83 | 0.29% | 12.69 / 12.66 / 12.65 | 12.66 | 0.18% |
| 8B 4w baseline | 101.30 / 93.82 / 90.96 | 93.82 | **5.60%** | 7.35 / 7.34 / 7.33 | 7.34 | 0.15% |
| 8B 4w wmma | 151.31 / 151.50 / 152.70 | 151.50 | 0.50% | 7.32 / 7.35 / 7.36 | 7.35 | 0.28% |
| 8B 8da4w baseline | 68.10 / 67.74 / 66.28 | 67.74 | 1.43% | 7.24 / 7.21 / 7.24 | 7.24 | 0.26% |
| 8B 8da4w wmma | 185.00 / 185.06 / 184.87 | 185.00 | 0.05% | 7.24 / 7.21 / 7.23 | 7.23 | 0.21% |

**Note on 8B/4w/baseline's 5.60% CoV**: the 3 reps decline monotonically (101.30 → 93.82 → 90.96),
not noise — this matches the documented thermal-throttling risk for tiled/non-coopmat configs under
sustained max-clock load (prior work found −19% to −27% cold→steady-state on 8B tiled configs; this
run shows a milder −10% drift across 3 back-to-back reps at true hardware-max clock). All other
combos, including 8B/wmma, stay flat (≤1.43% CoV) — coopmat configs don't show this drift, consistent
with prior findings. Reported honestly as raw reps + median, not blended into a single number.

## Speedup: wmma vs buffer baseline (our fully-optimized WMMA path vs non-WMMA buffer baseline)

| Model / Scheme | Baseline prefill (CoV) | WMMA prefill (CoV) | **Prefill speedup** | Baseline decode (CoV) | WMMA decode (CoV) | **Decode speedup** |
|---|---|---|---|---|---|---|
| 1B 4w | 610.43 (0.07%) | 846.63 (0.19%) | **×1.39** | 26.78 (0.12%) | 26.73 (0.21%) | ×1.00 |
| 1B 8da4w | 428.72 (0.04%) | 970.62 (0.20%) | **×2.26** | 25.66 (0.17%) | 25.59 (0.08%) | ×1.00 |
| 3B 4w | 224.73 (0.16%) | 313.25 (0.15%) | **×1.39** | 13.08 (0.36%) | 13.10 (0.20%) | ×1.00 |
| 3B 8da4w | 154.61 (0.05%) | 366.83 (0.29%) | **×2.37** | 12.63 (0.26%) | 12.66 (0.18%) | ×1.00 |
| 8B 4w | 93.82 (5.60%) | 151.50 (0.50%) | **×1.61** | 7.34 (0.15%) | 7.35 (0.28%) | ×1.00 |
| 8B 8da4w | 67.74 (1.43%) | 185.00 (0.05%) | **×2.73** | 7.24 (0.26%) | 7.23 (0.21%) | ×1.00 |

CoV = `stdev/mean`. Speedup = `median(wmma) / median(baseline)`.

## Cross-check against prior default-pin (509 MHz) data

Earlier this session, 1B buffer/wmma prefill was measured at the *default* 509 MHz pin:
4w = 458.99 tok/s, 8da4w = 531.40 tok/s. At this run's hardware-max pin (980 MHz, a 1.93× GPU clock
ratio over 509), those become 846.63 and 970.62 — actual scaling ratios of **1.845×** and **1.827×**,
slightly under the raw 1.93× clock ratio as expected (prefill isn't purely GPU-core-clock-bound; MIF/INT
and fixed overhead don't scale 1:1). No anomalous discrepancy — this cross-check passes.

## Artifacts

- Raw resumable results log: `results/maxclock_ab.jsonl` (36 rows, all `"ok": true`).
- Sweep driver stdout/log: `results/sweep_stdout.log`.
- Orchestrator script: `scripts/run_maxclock_ab.py` (adapted from
  `.shared-context/scripts/run_m5_full_sweep.py`).
