# STATUS: sarc-1.5-4070ti-fused-port

**2026-10-08 22:45 UTC. Candidate 1 (the fused attention kernel, profile `4070ti-fused1`, build `topic2` =
`6217da0a9`) is gated: `s2-c1` GATE_ACCEPTED, +11.64 % geomean over the parent. It is a one-pass kernel, so by
task section 6.4 no second port item is left on this card: the campaign closes with candidate 1 as its only
gated candidate. Running now, detached (`<artifact-dir>/queue/q05-roof-pristine.sh`): igpu-roofline `fast`
(fresh roofs), then the closing timed session `s3-pristine` (final stack against the pristine `dev/1.5` state,
7 repeats, about one hour).**

Next: read `s3-pristine`, write the closing report, `sarc/tools/check.sh --no-build`, push.

Blocking: nothing.

## Candidate 1 gated: `s2-c1` GATE_ACCEPTED 2026-10-08T22:39:57Z (all steps passed, plain pass)

Parent `6050b1287` with `4070ti-refine1` against `topic2` with `4070ti-fused1`; tok/s, median of 7 valid
interleaved runs per arm (`results/4070ti/sessions/s2-c1/{runs,summary}.csv`; medians, gains and geomean
recomputed from `runs.csv` with separate code: the same):

| cell | parent | candidate 1 | gain | prefill ms, parent -> candidate | ETDump dispatch total, ms | repeat spread, parent / candidate |
|---|---:|---:|---:|---|---|---|
| 1B 4w | 29681.2 | 35310.3 | +18.97 % | 69 -> 58 | 67.6 -> 56.3 | 1.47 % / 1.75 % |
| 1B 8da4w | 32507.9 | 39384.6 | +21.15 % | 63 -> 52 | 61.0 -> 50.2 | 1.61 % / 1.96 % |
| 3B 4w | 12962.0 | 14027.4 | +8.22 % | 158 -> 146 | 156.5 -> 143.7 | 0.63 % / 0.69 % |
| 3B 8da4w | 14948.9 | 16384.0 | +9.60 % | 137 -> 125 | 133.9 -> 123.9 | 2.88 % / 2.38 % |
| 8B 4w | 6023.5 | 6380.1 | +5.92 % | 340 -> 321 | 339.9 -> 320.6 | 0.88 % / 0.31 % |
| 8B 8da4w | 6989.8 | 7474.5 | +6.93 % | 293 -> 274 | 291.1 -> 271.8 | 0.68 % / 0.37 % |
| geomean | | | **+11.64 %** | | | |

84 timed and 36 next-token runs, all rc 0, all valid, none replaced; clock floor 2517 MHz applied, lowest
per-run median 2595 MHz, at least 2 clock samples per window; 0 slow loads of 120 (`raw/loads.csv`). On 1B one
step of the 1 ms timer is 1.7 to 1.9 % of the candidate's time; the ETDump totals (-16.6 %, -17.7 % dispatch
time) are the finer measure and agree.

Gate steps: 24 of 24 SDPA correctness passes (12 `extended`, 12 `full`) with 0 mismatches and `pairing=ok`,
every case served by the fused kernel (`sdpa-check.txt`: ACCEPT); unmodified `verify.sh` with the candidate
environment on the timed binaries: `verify.out` equals `s0-parent-verify` line by line, rates aside (34 lines;
verify-check ACCEPT, 0 findings); session-check ACCEPT, 0 findings, next token parent vs candidate SAME in all
24 rows (six cells x timed, real-text, check and unaligned prompt); 12 of 12 traced runs rc 0; env-check
ACCEPT. **No next-token item differs, so the gate recorded a plain pass and did not use the reference-error
rule.** The candidate does change the attention arithmetic, so that evidence was produced before the gate and
is reported all the same (below): the rule is met.

Where the gain comes from (warm ETDump, ms per 2048-token prefill, parent -> candidate 1;
`results/4070ti/sessions/s2-c1/trace/attention.csv`, `tools/attention_families.py`):

| cell | QK^T | softmax | attention x V | fused kernel | K / V copy | attention total | linear GEMM | everything else | attention share |
|---|---|---|---|---|---|---|---|---|---|
| 1B 4w | 4.17 -> 0 | 7.47 -> 0 | 5.49 -> 0 | 5.72 | 0.11 | 17.13 -> 5.84 | 34.28 -> 34.24 | 16.16 -> 16.27 | 25 % -> 10 % |
| 1B 8da4w | 4.15 -> 0 | 7.46 -> 0 | 5.28 -> 0 | 5.75 | 0.11 | 16.89 -> 5.86 | 27.65 -> 27.76 | 16.45 -> 16.57 | 28 % -> 12 % |
| 3B 4w | 8.46 -> 0 | 9.80 -> 0 | 8.14 -> 0 | 13.65 | 0.35 | 26.40 -> 14.00 | 96.90 -> 96.59 | 33.16 -> 33.16 | 17 % -> 10 % |
| 3B 8da4w | 8.24 -> 0 | 9.71 -> 0 | 8.07 -> 0 | 13.42 | 0.33 | 26.02 -> 13.75 | 73.22 -> 75.00 | 34.71 -> 35.14 | 19 % -> 11 % |
| 8B 4w | 13.00 -> 0 | 14.93 -> 0 | 11.62 -> 0 | 20.32 | 0.44 | 39.55 -> 20.76 | 239.40 -> 239.14 | 60.96 -> 60.69 | 12 % -> 6 % |
| 8B 8da4w | 12.50 -> 0 | 14.83 -> 0 | 11.51 -> 0 | 19.36 | 0.37 | 38.84 -> 19.73 | 181.12 -> 180.95 | 71.18 -> 71.08 | 13 % -> 7 % |

All of the gain is attention: the fused kernel removes 66 % of the attention time on 1B and 47 to 49 % on 3B
and 8B (the RX 7600 saw 76 %; its parent's softmax was a larger share). The copy pass costs 0.1 to 0.4 ms per
prefill. Linear kernels and everything else are unchanged.

Reference-error evidence (`results/4070ti/probe/fused1/`: `reference-error-rule.txt`, `REFERENCE_ERROR.json`,
`compare.csv`, `position/`), produced before the gate started:

- Criterion 1, error against the fp32 CPU reference on the same seeded inputs, both arms with `topic2`'s test
  binary, 0 mismatches in all 12 cases of both tiers for both arms. Production shapes (S = 2048), parent /
  candidate: 1B rms 2.101e-5 / 2.049e-5, maximum 9.14e-4 / 7.23e-4; 3B 2.068e-5 / 2.053e-5, 7.83e-4 / 7.06e-4;
  8B 2.057e-5 / 2.022e-5, 8.91e-4 / 7.91e-4: not larger in any production case. Outside the production shapes
  one of the two numbers is larger in 3 of 9 cases (3B S = 256 maximum +5 %, 8B S = 1024 at position 1024
  maximum +4 %, tiny S = 128 at position 64 rms +0.2 %); the full table is in the file. The two arms are equally
  accurate within a few per cent.
- Criterion 3, gross divergence on 41 real-text prompts (candidate default against parent default): top-1
  differs on 0 / 3 / 0 / 0 / 0 / 4 of 41 prompts (1B 4w / 1B 8da4w / 3B 4w / 3B 8da4w / 8B 4w / 8B 8da4w; the
  parent's own two linear arms differ on 0 / 6 / 0 / 1 / 0 / 2), mean KL at most 0.045 nat (limit 0.5), maximum
  KL 0.79 nat (1B 8da4w; the parent's own arms 1.67): none.
- Criterion 2, the position of the gate's unaligned item (prompt 0): all four arms pick the same token in every
  cell; `differing-items.txt` is empty.
- The last line of `compare.csv` prints the first decision's verdict ("outside twice the noise floor", in the
  three 4w cells, where the floor is the distance between two almost identical linear kernels: mean KL
  1.5e-5 to 3.5e-5 nat against the candidate's 5e-5 to 2.7e-4); the second decision replaced that test for
  arithmetic changes.

## Candidate 1 (`4070ti-fused1`): first runs, before the gate

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

Candidate 1: the table at the top. Baseline and A/A, session `s1-aa` (parent `6050b1287` against `topic1` = `35e3728c9`,
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
| `topic2` | `6217da0a9` (candidate 1) | spirv_golden PASS; first correctness passes, screen, reference-error evidence, gate `s2-c1` (accepted) |

All from exports of the commit and its 30 pinned submodules (`tools/mktree.sh`), `sarc/tools/build.sh --llama`
(and `--traced`) in `localhost/et-vk-build:rocky10` through the docker shim; provenance in
`<artifact-dir>/build/<tag>.src.txt`.

## Incidents

None. No device loss, no foreign GPU process, no aborted run.
