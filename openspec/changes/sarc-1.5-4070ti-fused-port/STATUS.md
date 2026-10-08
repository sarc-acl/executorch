# STATUS: sarc-1.5-4070ti-fused-port

**2026-10-08 20:57 UTC. Step 3 (candidate 1, the fused attention kernel, profile `4070ti-fused1`, build
`topic2` = `6217da0a9`). Its first correctness run is clean on every tier and the ungated looks are inside the
predicted band; nothing below is a gated number yet. Running now, detached
(`<artifact-dir>/queue/q04-c1-gate.sh`): the reference-error evidence (SDPA error of both arms with one test
binary, logits of 41 real-text prompts for four arms), then the gate `s2-c1` (`gate_sdpa.sh`: 24 SDPA passes,
`verify.sh`, timed session with 7 repeats, traces). Expected to end around 23:30 UTC.**

Next: read the gate; if accepted, close by task section 6.4 (candidate 1 is one-pass, so no second port item is
left): final verification on the committed head, the timed session against the pristine `dev/1.5` state, report.

Blocking: nothing.

## Candidate 1 (`4070ti-fused1`): first runs, not gated

Environment `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-fused1`, build `topic2` (spirv_golden PASS,
53 shipped variants).

Correctness, one pass per tier (`results/4070ti/first-correctness/`): `all` 4 of 4, `fused` 5 of 5, `peaked` 5
of 5, `extended` 8 of 8, `full` 4 of 4 cases PASSED with 0 mismatches; every case was served by the fused
kernel (`fused=sarc_dev_4070ti_sdpa_fused3sb_d64_t32x32g11s32rko` or `..d128_t16x64g11s32rko`, `qk=? softmax=?
av=?`, `pairing=ok`). Nothing in the kernel turned out to depend on lockstep execution: the first run on this
card was correct.

Error against the fp32 CPU reference, same seeded inputs (parent: `s0-parent-verify`, one pass; candidate: the
passes above; the gate evidence repeats this with one test binary for both arms):

| case | rms, parent / candidate 1 | maximum, parent / candidate 1 |
|---|---|---|
| 1B head configuration, S = 2048 (production) | 2.101e-5 / 2.049e-5 | 9.14e-4 / 7.23e-4 |
| 3B head configuration, S = 2048 (production) | 2.068e-5 / 2.053e-5 | 7.83e-4 / 7.06e-4 |
| 8B head configuration, S = 2048 (production) | 2.057e-5 / 2.022e-5 | 8.91e-4 / 7.91e-4 |
| 8B, S = 1024 at position 1024 | 9.018e-6 / 8.980e-6 | 6.94e-5 / **7.23e-5** |
| tiny, S = 128 at position 64 | 3.232e-5 / **3.238e-5** | 1.74e-4 / 1.46e-4 |
| 3B, S = 256 | 4.956e-5 / 4.875e-5 | 7.79e-4 / **8.20e-4** |
| the other six cases of `extended` | not larger | not larger |

Not larger on the three production shapes (criterion 1 of D3 names those); on three of the nine other shapes
one of the two numbers is larger by 0.2 to 5 %. The two arms are equally accurate to within a few per cent:
the parent already reduces its softmax in fp32.

Kernel level (`test_llama_microbench --sdpa`, us per layer at S = 2048, three rounds, profiles interleaved;
`results/4070ti/screens/sdpa-screen1.csv`). The fused number includes its K / V copy pass:

| model | stock (`dev/1.5`): QK^T + softmax + attention x V | parent `4070ti-refine1`: QK^T + softmax + attention x V = total | `4070ti-fused1`: fused + copy | fused / parent, per round |
|---|---|---|---|---|
| 1B | 3213 / 3234 / 3202 | 257 + 463 + 329 = 1048 / 1066 / 1060 | 382 / 387 / 388 | 0.364 / 0.363 / 0.366 |
| 3B | 3543 / 3547 / 3548 | 301 + 347 + 288 = 936 / 952 / 938 | 483 / 511 / 504 | 0.516 / 0.537 / 0.537 |
| 8B | 4708 / 4703 / 4712 | 397 + 462 + 350 = 1209 / 1227 / 1211 | 620 / 650 / 651 | 0.513 / 0.530 / 0.538 |

(The three-kernel split is round 1; the totals are the three rounds.) The fused kernel removes 64 % of the
attention time on 1B and 46 to 49 % on 3B and 8B, in every round.

Ungated end-to-end quick look (`quick_e2e.sh`, 2 runs per label, labels interleaved, same build;
`results/4070ti/screens/quick1.csv`): 1B +20.9 % / +21.6 %, 3B +8.6 % / +7.9 %, 8B +5.8 % / +6.9 % (4w / 8da4w),
geomean +11.75 %. A screen for deciding what to gate, not a result.

(Correction: the first version of this file, commit `e7a157bbe`, carried the time 20:30 UTC; it was written at
about 19:45 UTC.)

## Per-cell numbers against the parent

No gated candidate number yet. Baseline and A/A, session `s1-aa` (parent `6050b1287` against `topic1` = `35e3728c9`,
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
| `topic2` | `6217da0a9` (candidate 1) | spirv_golden PASS; first correctness passes, screen, gate `s2-c1` |

All from exports of the commit and its 30 pinned submodules (`tools/mktree.sh`), `sarc/tools/build.sh --llama`
(and `--traced`) in `localhost/et-vk-build:rocky10` through the docker shim; provenance in
`<artifact-dir>/build/<tag>.src.txt`.

## Incidents

None. No device loss, no foreign GPU process, no aborted run.
