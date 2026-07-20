# Secondary M5 EVT1 cross-check — `release13-node-threshold` branch (2026-07-11)

**Status**: single-run cross-check, NOT a replacement for this spec's canonical
3-run-mean numbers in `data-model.md` (measured on the **primary** M5 EVT1,
`dev`/`yanwen/dev-1.3` branch). Recorded here because it independently confirms
those numbers on a different device and a different branch, and because the
user asked to save it against this spec.

**Device**: secondary M5 EVT1, `ssh xgpusw-debug08`, `ANDROID_SERIAL=00000bf70c579c33`
(see workspace memory `m5-evt1-secondary-debug08`).

**Branch/worktree**: `release13-node-threshold/` (`yanwen/release13-node-threshold`,
off `release/1.3` — no WMMA/coopmat on this branch; texture PTEs are the stock
T-tiled op path by construction, so no separate dispatch-confirmation step was
needed the way `data-model.md`'s primary-device measurement required).

**Driver**: md5 `c9861e9906d03fa2c7d48b804e1a1c80` (= `f14c51b6f8`, known-good,
matches primary M5 EVT1) — verified before the session, re-verified unchanged
after the final run. No reflash needed.

**Clocks**: pinned 509/2730/663 MHz (`pin_freqs.sh`), verified before and after
every run — held steady, no drift across the whole session.

**Workload**: 2048-token prefill + 1024-token decode (`_ctx3072.pte`), single
run per config (not a 3-run mean — treat CoV as unknown; use `data-model.md`'s
primary-device 3-run means as the trusted numbers for reporting).

## Results

| Model | Quant | Prefill tok/s | Decode tok/s | Model load | Inference | Total wall time |
|---|---|---|---|---|---|---|
| llama3_2_1b | 4w | 310.021 | 14.337 | 0.79s | 77.96s | 156.34s |
| llama3_2_1b | 8da4w | 222.078 | 13.7809 | 1.40s | 83.45s | 167.69s |
| llama3_2_3b | 4w | 112.62 | 7.18995 | 1.73s | 160.47s | 322.19s |
| llama3_2_3b | 8da4w | 79.6887 | 6.81537 | 2.93s | 175.80s | 353.73s |
| llama3_1_8b | 4w | 51.4832 | 3.96203 | 5.61s | 297.98s | 601.03s |
| llama3_1_8b | 8da4w | 35.1558 | 3.84756 | 6.00s | 324.14s | 653.54s |

## Cross-check vs this spec's canonical (primary-device, `dev` branch) numbers

| Model | Quant | This run (secondary, `release13-node-threshold`) | `data-model.md` canonical (primary, `dev`) | Delta |
|---|---|---|---|---|
| 1B | 4w | 310.02 | 312.7 (`RESULTS-SUMMARY.md` anchor) | -0.9% |
| 1B | 8da4w | 222.08 | 222.30 | -0.1% |
| 3B | 4w | 112.62 | 112.5 (anchor) | +0.1% |
| 3B | 8da4w | 79.69 | 79.83 | -0.2% |
| 8B | 4w | 51.48 | 51.4 (anchor) | +0.2% |
| 8B | 8da4w | 35.16 | 35.17 | -0.03% |

All deltas are well within normal run-to-run variance (this run is a single
sample vs. the canonical 3-run means) — strong agreement, no discrepancy
worth investigating.

## Anomalies

- Initial 3B/4w attempt was killed by an SSH-side `timeout 300` wrapper before
  finishing (the run itself simply took ~320s, longer than the timeout) —
  cosmetic, not a device/driver issue. Retried with a longer timeout and
  completed cleanly.
- No sgpu watchdog kills on any of the 6 runs (all used
  `ET_VK_EXECUTE_NODE_THRESHOLD=16`), no driver drift, no segfaults.

## Raw logs

- `.artifacts/e2e-2026-07-11-secondary/llama3_2_1b_4w_texture_2048p1024d.log`
- `.artifacts/e2e-2026-07-11-secondary/llama3_2_1b_8da4w_texture_2048p1024d.log`
- `.artifacts/e2e-2026-07-11-secondary/llama3_2_3b_4w_texture_2048p1024d.log`
- `.artifacts/e2e-2026-07-11-secondary/llama3_2_3b_8da4w_texture_2048p1024d.log`
- `.artifacts/e2e-2026-07-11-secondary/llama3_1_8b_4w_texture_2048p1024d.log`
- `.artifacts/e2e-2026-07-11-secondary/llama3_1_8b_8da4w_texture_2048p1024d.log`
- Command log: `.artifacts/cmd-log-2026-07-11.sh`
