# specs/037: M51 deep tile sweep → 4w/8da4w prefill-speedup table

Overnight, unattended follow-on to specs/036-portable-device-sweep, run on
`dev/executorch` (branch `yanwen/dev-1.3`) against the currently-shipped
coopmat shaders (no shader source changes here).

## Goal

A deeper Optuna tile sweep (budget 180 vs the shipped budget-60 pass) for both
`q4gsw` (4w) and `dq8ca` (8da4w), at **hardware-max clock** (980/5333/934
GPU/MIF/INT), validated at the size that matters (8B), producing the single
required deliverable: a 6-row table of **prefill tok/s speedup over the stock
T-tiled baseline** for {1B, 3B, 8B} × {4w, 8da4w}.

The sweep is instrumental; the table is the goal. Ordered **sweep → validate →
matrix** so the table always gets produced (best validated config, else the
current shipped default — "no improvement" is a normal, valid outcome).

## Decisions (confirmed with user 2026-07-24)

- Metric: **prefill tok/s only** (`--max_new_tokens≈8` in the final matrix).
- Baseline: **T-tiled** = stock texture PTE, no coopmat env.
- Clocks: **max pin** 980000/5333000/934000 (GPU/MIF/INT), for sweep and matrix.
- Driver drift: verify md5 (`c9861e9906d03fa2c7d48b804e1a1c80` = `f14c51b6f8`)
  at every phase boundary; **auto-reflash** from
  `/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so` on drift, then continue.

## Phases

0. **Pre-flight** — driver md5 verify/reflash, max-clock pin, Android rebuild
   (reconfigure so the tsweep YAML glob is picked up), correctness gate +
   coherence check, PTE staging.
1. **Deep sweep** (`../036-portable-device-sweep/scripts/sweep.py
   --slug-suffix maxclk`) — both shaders, budget 180, early-stop 40,
   8 finalists, 3 reps, `--remote android`; dq8ca adds
   `--quirk no_int8_wmma_sg32`. Resumable via the Optuna journal.
2. **Cross-size validation** (`scripts/pick_winner.py`) — re-confirms the
   top-3 1B finalists (+ shipped default, unconditionally) on 8B and 3B;
   winner = whichever wins at 8B, else shipped default.
3. **Final matrix** (`scripts/final_matrix.py`, adapted from
   `../036-m51-maxclock-coopmat-ab/scripts/run_maxclock_ab.py`) — 36 runs:
   {1B,3B,8B} × {4w,8da4w} × {baseline=T-tiled, coopmat=winner-variant} × 3
   reps, prefill-only, maxpin, `ET_VK_EXECUTE_NODE_THRESHOLD=32` on every run.
4. **Report** (`scripts/make_report.py`) — `results/report.md`, the 6-row
   speedup table.

`scripts/orchestrate.sh` chains phases 0→4 and is meant to be launched
detached (nohup) for the overnight run; every phase is independently
resumable so a crash in one doesn't lose the others.

## Caveats

- The dq8ca tsweep shader is dbuf2; the shipped production shader is dbuf4.
  If a *new* dq8ca winner is later shipped (not part of this run), it needs
  the protocol.md §6 dbuf re-verify first.
- Rank flips between 1B and 8B are documented (protocol.md §5) — the winner
  is chosen at 8B specifically for this reason.
- `measure_android.Session` gained an `extra_env` constructor param (small,
  additive change, see `../036-portable-device-sweep/scripts/measure_android.py`)
  so Phase 2's 8B validation can set `ET_VK_EXECUTE_NODE_THRESHOLD=32` — the
  watchdog workaround `run_maxclock_ab.py` already applies for 8B — without
  which an 8B prefill run risks a silent watchdog crash mid-validation.
