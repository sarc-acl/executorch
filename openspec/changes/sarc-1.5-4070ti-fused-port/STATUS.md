# STATUS: sarc-1.5-4070ti-fused-port

**2026-10-08 20:45 UTC. Steps 1 and 2 done (baseline, A/A, snapshots, hook commit and its controls). Step 3
(candidate 1, the fused attention kernel, profile `4070ti-fused1`) is under way. Running now, detached
(`<artifact-dir>/queue/q03-c1-first.sh`): the build of `topic2` (`6217da0a9`), then one first correctness pass
per tier (`all`, `fused`, `peaked`, `extended`, `full`), a kernel-level SDPA screen (stock, `4070ti-refine1`,
`4070ti-fused1`, 3 rounds) and an ungated end-to-end quick look. Candidate 1 has not run on the GPU yet.**

Next: read the first correctness passes; if clean, the reference-error evidence (SDPA error of both arms, the
41-prompt logits comparison), then the gate (`gate_sdpa.sh`) with its timed session.

Blocking: nothing.

(Correction: the first version of this file, commit `e7a157bbe`, carried the time 20:30 UTC; it was written at
about 19:45 UTC.)

## Per-cell numbers against the parent

No candidate measured yet. Baseline and A/A, session `s1-aa` (parent `6050b1287` against `topic1` = `35e3728c9`,
the hook commit; both arms `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1`; tok/s, median of 5
valid interleaved runs per arm; `results/4070ti/sessions/s1-aa/`):

| cell | parent | topic1, same profile | ratio | expected (task file) | parent against expected | repeat spread, parent / topic1 |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 29681.2 | 29681.2 | 1.0000 | 29681 | +0.00 % | 2.90 % / 1.47 % |
| 1B 8da4w | 32507.9 | 33032.3 | 1.0161 | 33032 | -1.59 % | 1.61 % / 3.23 % |
| 3B 4w | 12962.0 | 12962.0 | 1.0000 | 12881 | +0.63 % | 0.63 % / 0.63 % |
| 3B 8da4w | 14948.9 | 15058.8 | 1.0074 | 14949 | +0.00 % | 2.17 % / 2.16 % |
| 8B 4w | 6023.5 | 6023.5 | 1.0000 | 5988 | +0.59 % | 0.29 % / 0.29 % |
| 8B 8da4w | 6989.8 | 6989.8 | 1.0000 | 6954 | +0.51 % | 0.34 % / 0.34 % |

- Baseline: every cell within 1.6 % of the expected number (limit 3 %).
- A/A: geomean 1.0039 (limit 1 % from 1), largest cell difference +1.61 % (limit 2 %): 1B 8da4w, where the
  two medians are 63 ms and 62 ms, one step of the runner's 1 ms timer. 60 timed and 36 next-token runs, all rc 0
  and valid, none replaced; next token SAME in all 24 rows; 0 slow loads of 96 (`raw/loads.csv`);
  `gate_check.py session --calibration`: ACCEPT, 0 findings.
- Calibration (`tools/thresholds.txt`, `results/4070ti/clkmin.json`): lowest per-run median clock 2595 MHz, no
  ramp run, **clock floor 2517 MHz**; at least 3 clock samples in every window. **Repeats: 7**, by the rule
  fixed beforehand (an A/A cell further than 1.0 % from 1, and one arm's spread above 3.0 %: both are the 1B
  8da4w timer step). The gate tools take the repeat count from `thresholds.txt`.

## Parent snapshots (unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, build `parent`)

| snapshot | environment | result |
|---|---|---|
| `s0-parent-verify` | the parent's (`4070ti-refine1`) | correctness rc 0, `linear 4w rc=1` and `linear 8da4w rc=1` (24 confirmed + 24 `unexpected_coopmat`, as in the first campaign), 12 of 12 production-diff ALL PASSED, the four default-vs-tiled items SAME, decode 31 tokens, 22 runner calls rc 0; `gate_check.py verify` ACCEPT |
| `s0-pristine-verify` | none | the same statuses; ACCEPT |

SDPA correctness on the parent, one pass per tier (recorded): 0 mismatches in every case with the profile
(kernels `qk ..4070ti_df_t64x64k32g11s32nf` / `..pk_t64x128k32g42s32nf`, softmax `..4070ti_nzf`,
`av ..ml_t32x64k32g42s32` / `..ml_t64x128k32g42s32`).

## Hook control (owner decision 2026-10-05, D4 conditions), build `topic1` = `35e3728c9`

| condition | result |
|---|---|
| nothing selected: `verify.sh` with no environment against `s0-pristine-verify` | `s1-ctl-noenv`: CONTROL_SAME, 34 lines equal line by line (rates aside), `control.diff` empty, verify-check ACCEPT |
| the parent's profile: `verify.sh` with `4070ti-refine1` against `s0-parent-verify` | `s1-ctl-parent`: CONTROL_SAME, 34 lines, `control.diff` empty, verify-check ACCEPT |
| `spirv_golden.py` | PASS, 53 shipped variants unchanged, on `parent` and on `topic1` (the Orin's shared 8da4w kernel included) |
| `test_sarc_select` (`sarc/tools/check.sh --no-build`, on the tree with candidate 1's dev-zone code) | PASS: 1240 checks and 31 rows with the release tables alone, 1536 checks and 33 rows with the dev zone, the same counts as at the end of the first campaign |
| timed, same profile in both arms | the A/A session above: geomean 1.0039 |

## Coordinator hold

`<artifact-dir>/HOLD` (coordinator) and `<artifact-dir>/HELD` (this campaign), `tools/common.sh: hold_point`.
Smallest unit: one cell of a timed session (about 6 minutes on 8B with 7 repeats), one `verify.sh` (about 5
minutes), one SDPA correctness pass (up to 3 minutes for the `full` tier), one traced run, one screen run, one
build (about 6 minutes). While held, a timed session releases the device lock and takes it again afterwards.
Tested 2026-10-08 19:48 UTC through the same code path with the test name `HOLD-TEST`: `HELD-TEST` appeared
with the unit's name, the lock was free while held, and after the file was removed the unit resumed with the
lock; `HELD-TEST` was removed.

## Built

| tag | commit | note |
|---|---|---|
| `parent` | `6050b1287` | spirv_golden PASS (53 shipped variants) |
| `topic1` | `35e3728c9` (hook) | spirv_golden PASS; hook controls, A/A arm |
| `topic2` | `6217da0a9` (candidate 1) | building |

All from exports of the commit and its 30 pinned submodules (`tools/mktree.sh`), `sarc/tools/build.sh --llama`
(and `--traced`) in `localhost/et-vk-build:rocky10` through the docker shim; provenance in
`<artifact-dir>/build/<tag>.src.txt`.

## Incidents

None. No device loss, no foreign GPU process, no aborted run.
