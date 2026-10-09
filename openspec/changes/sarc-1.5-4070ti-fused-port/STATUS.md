# STATUS: sarc-1.5-4070ti-fused-port

**2026-10-09 03:20 UTC. CLOSED again after review round 1. `4070ti-fused1` on build `topic3` (`ed8b5af91`, the
committed branch, no local patch): gate `s4-c1` GATE_ACCEPTED, +11.49 % geomean over the parent; closing
session `s5-pristine`, +63.28 % over the pristine `dev/1.5` state. One gated candidate, no second port item on
this card (task section 6.4): closed by N3.**

Running now: nothing. Blocking: nothing. No decision is needed from the owner.

## What the review found and what was done about it

| finding | what was done | evidence |
|---|---|---|
| 1. the gate ran 12 passes of `extended` and of `full`, none of `all` | `gate_sdpa.sh` runs and `gate_check.py sdpa` requires 12 passes of each of `all`, `extended`, `full`; the directory records the hashes of the test binary and runner library, the environment and every pass's exit status | `results/4070ti/sessions/s4-c1/sdpa-correctness/` (`IDENTITY.txt`, `exit-status.txt`, 36 logs), `sdpa-check.txt` |
| 2. no timed run recorded the driver's thermal-throttle reasons | the sampler reads `clocks_event_reasons.sw_thermal_slowdown`, `.hw_thermal_slowdown`, `.hw_slowdown` and the raw mask with every 20 ms clock sample; a timed run with a thermal reason in any sample, or with fewer than 2 such samples, is invalid (`runrow.py`, `gate_check.py session`) | columns `thr_n`, `thr_thermal_n`, `thr_hw_slowdown_n`, `thr_masks` of `runs.csv` in `s4-aa`, `s4-c1`, `s5-pristine` |
| 3. the shared test `test_llama_microbench.cpp` was edited in place (R3) | the file is the parent's plus eight blocks delimited by `// >>> 4070ti-fused ...` / `// <<< 4070ti-fused ...`: 99 added lines, 0 removed or modified (`git diff 6050b1287 HEAD` on the file). No owner exception was asked for | commit `ed8b5af91` |
| 4. `proposal.md` misdescribed the tools | corrected (tools section) | `proposal.md` |

Because test source changed, everything was built again (`topic3`), gated again (`s4-c1`) and timed again
(`s4-aa`, `s5-pristine`). Between `topic2` and `topic3` only that test file differs outside this directory
(`git diff --name-status 6217da0a9 ed8b5af91 -- . ':!openspec'`); the three candidate SPIR-V files are
byte-identical in the two builds. Thresholds were not re-calibrated: clock floor 2517 MHz and 7 repeats stay as
committed before the first candidate.

**The sessions on `topic2` (`s1-aa`, `s2-c1`, `s3-pristine`) have no record of throttle reasons. They are kept
as evidence with that limitation and are not the reported result; the absence of throttling in them is not
inferred from temperature or clock.** `s2-c1` also lacks the 12 `all` passes.

## Closing numbers (build `topic3`)

tok/s, median of 7 valid interleaved runs per arm (`results/4070ti/sessions/{s4-c1,s5-pristine}/`; every
median, gain, spread, geomean and validity flag recomputed from `runs.csv`, the per-run logs and the clock
samples with separate code: the same):

| cell | pristine `dev/1.5` (`s5-pristine`) | published | parent `4070ti-refine1` (`s4-c1`) | `4070ti-fused1` (`s4-c1` / `s5-pristine`) | over the parent | over pristine | over published |
|---|---:|---:|---:|---|---:|---:|---:|
| 1B 4w | 19883.5 | 19692.3 | 29681.2 | 35310.3 / 35929.8 | +18.97 % | +80.70 % | +82.46 % |
| 1B 8da4w | 21113.4 | 20898.0 | 32507.9 | 39384.6 / 39384.6 | +21.15 % | +86.54 % | +88.46 % |
| 3B 4w | 8752.1 | 8752.1 | 12962.0 | 14027.4 / 14027.4 | +8.22 % | +60.27 % | +60.27 % |
| 3B 8da4w | 9752.4 | 9660.4 | 14948.9 | 16254.0 / 16254.0 | +8.73 % | +66.67 % | +68.25 % |
| 8B 4w | 4511.0 | 4491.2 | 6023.5 | 6380.1 / 6400.0 | +5.92 % | +41.88 % | +42.50 % |
| 8B 8da4w | 5056.8 | 5031.9 | 6989.8 | 7474.5 / 7501.8 | +6.93 % | +48.35 % | +49.08 % |
| geomean | | | | | **+11.49 %** | **+63.28 %** | +64.34 % |

The gains over the published numbers use the unrounded medians of `sarc-1.5-e2e-benchmark/results/cells.csv`.
Every gain is outside the +-2 % noise band. The parent arm is within 1.59 % of the task's expected numbers in
every cell (limit 3 %), the pristine arm within 1.03 % of the published ones. The first gate on `topic2`
(`s2-c1`, without throttle reasons) read +11.64 %; five cells have the same medians and 3B 8da4w is one timer
step lower (16254.0 against 16384.0).

### Gate `s4-c1`: GATE_ACCEPTED 2026-10-09T02:38:27Z, plain pass

Parent `6050b1287` with `4070ti-refine1` against `topic3` with `4070ti-fused1`.

| cell | parent | candidate 1 | gain | prefill ms | ETDump dispatch total, ms | repeat spread, parent / candidate |
|---|---:|---:|---:|---|---|---|
| 1B 4w | 29681.2 | 35310.3 | +18.97 % | 69 -> 58 | 67.6 -> 56.3 | 2.90 % / 1.75 % |
| 1B 8da4w | 32507.9 | 39384.6 | +21.15 % | 63 -> 52 | 61.2 -> 50.0 | 1.61 % / 1.96 % |
| 3B 4w | 12962.0 | 14027.4 | +8.22 % | 158 -> 146 | 156.0 -> 142.6 | 0.00 % / 0.69 % |
| 3B 8da4w | 14948.9 | 16254.0 | +8.73 % | 137 -> 126 | 136.0 -> 123.1 | 1.46 % / 2.40 % |
| 8B 4w | 6023.5 | 6380.1 | +5.92 % | 340 -> 321 | 340.1 -> 320.2 | 0.29 % / 0.31 % |
| 8B 8da4w | 6989.8 | 7474.5 | +6.93 % | 293 -> 274 | 291.2 -> 274.2 | 1.03 % / 0.73 % |
| geomean | | | **+11.49 %** | | | |

- SDPA correctness with the candidate environment on `topic3`'s test binary (sha256 `61d29fdb...`, equal to
  the build's; runner library `cca20e61...`, the timed one): 12 passes of `all` (4 cases each), 12 of
  `extended` (8), 12 of `full` (4): 192 of 192 cases PASSED, `mismatches=0`, `pairing=ok` on every case, every
  case served by the fused kernel (d64 variant 108 times, d128 variant 84 times), 36 of 36 exit statuses 0;
  `sdpa-check.txt`: ACCEPT, 0 findings. Before the gate, one pass each of `fused` (5 of 5) and `peaked` (5 of 5).
- Unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate environment on the
  timed binaries: `verify.out` equals `s0-parent-verify` in all 34 lines once the 14 rates are removed
  (checked with `diff` separately from the gate checker); correctness rc 0 / 0, linear rc 1 / 1 in both schemes
  (24 confirmed + 24 `unexpected_coopmat`, as the parent); 22 runner calls rc 0; verify-check ACCEPT.
- Timed session (01:54 to 02:30 UTC): 84 timed and 36 next-token runs, all rc 0, all valid, none replaced;
  arms alternate parent-first on odd repeats; lowest per-run median clock 2595 MHz against the floor 2517; at
  least 2 clock samples per window; start temperatures 45 to 50 C, maximum 67 C. **1 slow load of 120**
  (`loads.csv`, by `load_report.py`'s definition: the time the clock was below 1500 MHz between the first and
  the last sample at or above 2000 MHz, slow at 1.0 s or more): `prefill-8b-8da4w-parent-r1`, the first process
  of the cell, has a low-clock gap of 3.005 s in the middle of the process. Its clock first reached 2000 MHz
  0.47 s after sampling began, fell below 2000 MHz at 3.97 s and returned at 6.98 s; the samples span 7.47 s and
  the runner's own model load took 6.32 s. The model file was 93 % cached before the warming pass and 100 %
  after it. The run ended
  rc 0 with 15 clock samples in its window at a median of 2760 MHz and no thermal reason; it is valid under the
  fixed criteria and is kept, classified slow as recorded.
  **Throttle reasons: at least 48 samples per timed run, a thermal reason (`sw_thermal_slowdown`,
  `hw_thermal_slowdown`) active in 0 samples of every run, `hw_slowdown` in 0.** The raw masks seen are
  `0x1` (idle), `0x4` (the power cap, set under load on this card by design, L16), `0x400` and `0x404`; the
  driver's three named flags are what the validity rule reads.
- Next token parent vs candidate: SAME in 24 of 24 rows (six cells x timed, real-text, check and unaligned
  prompt). **No item differs, so the gate recorded a plain pass and did not use the reference-error rule**; the
  evidence for that rule is reported below all the same, because the candidate changes the arithmetic.
- 12 of 12 traced runs rc 0; env-check ACCEPT; `spirv_golden.py` on `topic3`: PASS (53 shipped variants), and
  every SPIR-V file of the parent build is byte-identical in `topic3`.

Where the gain comes from (warm ETDump, ms per 2048-token prefill, parent -> candidate;
`results/4070ti/sessions/s4-c1/trace/attention.csv`, `tools/attention_families.py`):

| cell | QK^T | softmax | attention x V | fused kernel | K / V copy | attention total | linear GEMM | everything else | attention share |
|---|---|---|---|---|---|---|---|---|---|
| 1B 4w | 4.18 -> 0 | 7.47 -> 0 | 5.42 -> 0 | 5.71 | 0.11 | 17.07 -> 5.82 | 34.30 -> 34.21 | 16.19 -> 16.29 | 25 % -> 10 % |
| 1B 8da4w | 4.15 -> 0 | 7.46 -> 0 | 5.41 -> 0 | 5.70 | 0.11 | 17.02 -> 5.81 | 27.66 -> 27.67 | 16.50 -> 16.56 | 28 % -> 12 % |
| 3B 4w | 8.44 -> 0 | 9.80 -> 0 | 8.17 -> 0 | 13.58 | 0.34 | 26.40 -> 13.92 | 96.58 -> 95.74 | 33.06 -> 32.94 | 17 % -> 10 % |
| 3B 8da4w | 8.35 -> 0 | 9.70 -> 0 | 8.09 -> 0 | 13.34 | 0.33 | 26.14 -> 13.67 | 74.76 -> 74.38 | 35.11 -> 35.01 | 19 % -> 11 % |
| 8B 4w | 13.01 -> 0 | 14.92 -> 0 | 11.63 -> 0 | 20.29 | 0.44 | 39.56 -> 20.73 | 239.58 -> 238.70 | 60.96 -> 60.79 | 12 % -> 6 % |
| 8B 8da4w | 12.51 -> 0 | 14.83 -> 0 | 11.49 -> 0 | 19.40 | 0.42 | 38.83 -> 19.82 | 181.16 -> 181.27 | 71.21 -> 73.08 | 13 % -> 7 % |

All of the gain is attention: the fused kernel removes 66 % of the attention time on 1B and 47 to 49 % on 3B
and 8B. The copy pass costs 0.1 to 0.4 ms per prefill. The linear kernels are unchanged.

Kernel level (`test_llama_microbench --sdpa`, us per layer at S = 2048, three interleaved rounds on `topic3`,
`results/4070ti/screens/sdpa-screen2.csv`; the fused number includes its copy pass):

| model | stock `dev/1.5` | parent: QK^T + softmax + attention x V (round 1) = total per round | `4070ti-fused1` per round | fused / parent per round |
|---|---|---|---|---|
| 1B | 3218 / 3233 / 3189 | 259 + 463 + 341 = 1063 / 1059 / 1065 | 381 / 369 / 381 | 0.358 / 0.348 / 0.358 |
| 3B | 3553 / 3541 / 3545 | 303 + 347 + 292 = 942 / 936 / 935 | 510 / 488 / 509 | 0.542 / 0.521 / 0.545 |
| 8B | 4705 / 4701 / 4711 | 399 + 463 + 353 = 1215 / 1213 / 1211 | 654 / 619 / 647 | 0.539 / 0.510 / 0.534 |

Reference-error evidence on `topic3` (`results/4070ti/probe/fused1-topic3/`), produced before the gate:

- Criterion 1 (D3), error against the fp32 CPU reference, both arms with `topic3`'s test binary, 0 mismatches in
  all 12 cases for both arms. Production shapes (S = 2048), parent / candidate: 1B rms 2.101e-5 / 2.049e-5,
  maximum 9.14e-4 / 7.23e-4; 3B 2.068e-5 / 2.053e-5, 7.83e-4 / 7.06e-4; 8B 2.057e-5 / 2.022e-5, 8.91e-4 /
  7.91e-4: not larger in any production case. Outside the production shapes one metric is larger in 3 of 9
  cases (3B S = 256 maximum +5 %, 8B S = 1024 at position 1024 maximum +4 %, tiny at position 64 rms +0.2 %).
  The per-case lines are identical to those of `topic2` (same seeded inputs, same shaders).
- Criterion 3, 41 real-text prompts, candidate default against parent default: top-1 differs on 0 / 3 / 0 / 0 /
  0 / 4 of 41 (1B 4w / 1B 8da4w / 3B 4w / 3B 8da4w / 8B 4w / 8B 8da4w), mean KL at most 0.045 nat (limit 0.5):
  no gross divergence.
- Criterion 2: all four arms pick the same token at the gate's positions; `differing-items.txt` lists none.
- `compare.csv` ends with the first decision's verdict (outside twice the floor of the parent's two linear arms
  in the three 4w cells); D3 replaced that test for arithmetic changes. `ref_error_rule.py`: MET.

### A/A re-check `s4-aa` (parent against `topic3`, both `4070ti-refine1`, throttle reasons recorded)

84 timed runs, all valid, 0 thermal samples. Parent / `topic3`: 29681.2 / 29681.2, 32507.9 / 32507.9, 12962.0 /
12880.5, 14948.9 / 15058.8, 6023.5 / 6023.5, 6966.0 / 6966.0; ratios 1, 1, 0.9937, 1.0074, 1, 1; geomean 1.0002.
The insert-only test blocks and the hook leave the parent's profile where it was.

### Hook controls on `topic3`

`s4-ctl-noenv` (no environment, against `s0-pristine-verify`) and `s4-ctl-parent` (`4070ti-refine1`, against
`s0-parent-verify`): CONTROL_SAME, 34 lines equal with rates removed, verify-check ACCEPT, in both.

### Closing session `s5-pristine` (02:40 to 02:54 UTC)

Parent build with no environment against `topic3` with `4070ti-fused1`: 84 timed and 36 next-token runs, all
rc 0 and valid, none replaced; lowest per-run median clock 2580 MHz; 0 thermal samples in every run (masks as
above); 0 slow loads; next token SAME in 24 of 24 rows against the stock attention kernels as well;
`gate_check.py session`: ACCEPT, 0 findings. It is a timed session, not a gate.

### Roofs

igpu-roofline `fast`, run `roofline/2026-10-08-fast` (22:40 to 23:05 UTC, driver 615.71.09): matrix fp16
182.8 TFLOP/s, fp16 with fp32 accumulator 92.3, int8 369.2 TOP/s; DRAM read 714, write 641, copy 646 GB/s.
From `s4-c1`'s traces (`results/4070ti/roofline/{fused,gemm}-roof-s4-c1.txt`): the fused kernel runs at 53 %
(d64, 48.9 to 49.0 TFLOP/s) and 59 to 63 % (d128, 54.8 to 58.5; rates from the unrounded dispatch times) of the
fp16 -> fp32 matrix roof; the unchanged
linear kernels at 64 to 66 % (4w, 116 to 121 TFLOP/s) and 39 to 43 % (8da4w, 144 to 158 TOP/s) of theirs.

### Final checks on the committed tree (2026-10-09 03:05 UTC)

`sarc/tools/check.sh --no-build`:

```
== 1 zone rule vs origin/release/1.5
== 2 twin wrappers
== 3 test_sarc_select
test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
[sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
test_sarc_select: PASS (1536 checks, 33 rows, 204 candidates, dev zone linked, unverified on)
check.sh: PASS
```

The two CPU selector executables of step 3 are retained for the reviewer, who may not build
(`<artifact-dir>/check-select/`, 2026-10-09 03:47 UTC). `check.sh` deletes its own, so the unmodified script was
run once more with `CXX=<artifact-dir>/check-select/cxx-keep.sh`, a wrapper that calls `c++` with the arguments
it is given and copies the output file; `check.sh` reads `CXX` from the environment and is unchanged (byte-identical
to the parent's). The run printed the same seven lines and PASS (`check.out`).

| file | content |
|---|---|
| `test_sarc_select.rel` | release tables alone; sha256 `954efa2fef42a363394c7d5707a25c67fefb19928d98dd08d5b3ad992fb181a5` |
| `test_sarc_select.dev` | with the dev zone; sha256 `76bede2257eec0e261e0d7f737c9a7cb53891bd9f49c106600a73ae55ae20ddf` |
| `compile-commands.txt` | the two exact compiler commands and their working directory |
| `yamls.txt` | the 34 yaml arguments, in `check.sh`'s order |
| `run.sh` | runs both executables as step 3 does (`ET_VK_SARC_UNVERIFIED=1` for the second), no compilation: `bash <artifact-dir>/check-select/run.sh` |
| `PROVENANCE.txt` | time, compiler (`c++` 15.2.0), hashes, working-copy head `88bedad43` with a clean tree; its source equals the measured commit `ed8b5af91` outside `openspec/` |

Run again from the retained files: 1240 checks, 31 rows, 0 candidates (release); 1536 checks, 33 rows, 204
candidates (dev zone); both rc 0.

It does not report the release-zone hook: `impl/sarc/` is inside the zones it checks and `impl/SDPA.cpp` is
listed in `sarc/HOOKS`. Files changed against the parent outside the dev zone and this directory: the four of
the hook (`impl/SDPA.cpp`, `impl/sarc/SdpaCoopmat.cpp`, `impl/sarc/SdpaCoopmat.h`, `impl/sarc/Select.h`, commit
`35e3728c9`, owner decision of 2026-10-05, D4.3) and nothing else; nothing under `sarc/`. The measured build
`topic3` is commit `ed8b5af91`; later commits change only this directory. `tools/test_gate_check.py`: 48 tests
pass.

All 920 `llama_main` calls of the campaign ended with rc 0 (176 in eight `verify.sh` runs, 696 in six
sessions, 24 traced, 24 in the quick look): no runner abort, no device loss, no foreign GPU process. One of the 696 session runs was a slow load (in
`s4-c1`, above); the other five sessions record none. The kernel
log of this boot (since 2026-10-07 16:45 UTC, read with `journalctl -k`) has no `Xid` line.

**Everything below this line is the record of the state before the review (build `topic2`, sessions without
throttle reasons) and is not the reported result.**

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

## Per-cell numbers against the parent

Candidate 1: the tables above. Baseline and A/A, session `s1-aa` (parent `6050b1287` against `topic1` = `35e3728c9`,
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
| `topic2` | `6217da0a9` (candidate 1) | spirv_golden PASS; first correctness passes, screen, reference-error evidence, gate `s2-c1` (accepted; superseded by `topic3` after the review) |
| `topic3` | `ed8b5af91` (candidate 1, test changes as insert-only blocks) | spirv_golden PASS; hook controls `s4-ctl-*`, A/A `s4-aa`, screen 2, reference-error evidence, gate `s4-c1` (accepted), closing session `s5-pristine` |

All from exports of the commit and its 30 pinned submodules (`tools/mktree.sh`), `sarc/tools/build.sh --llama`
(and `--traced`) in `localhost/et-vk-build:rocky10` through the docker shim; provenance in
`<artifact-dir>/build/<tag>.src.txt`.

## Incidents

None on the device: no device loss, no foreign GPU process, no aborted run. Review round 1 reopened the
campaign once (see the top of this file).

Correction kept on record: the first version of this file (commit `e7a157bbe`) carried the time 20:30 UTC; it
was written at about 19:45 UTC.
