# sarc-1.5-b580-fused-port: status

**2026-10-09 15:26 UTC — closing. Final stack `b580-fused1` on the committed head: **+9.12 % geomean** over the
parent (`s6-final`, `GATE_PASS`), **+72.05 %** over the pristine parent (`s7-pristine`), +76.2 % over the published
numbers; candidate 2 +0.08 % (`s3c-c2`, not a gain, not adopted). All numbers are measured on **pipelines created
without the full-subgroups flag (F1)**: by the owner decision of 15:25 UTC ("F1: C") F1 is a known defect of the
release zone shared with every cooperative-matrix pipeline of every device, it is not changed in this campaign,
and nothing is rebuilt, re-gated or re-timed for it: see "Known defect: F1". A second finding of the last review
(the correctness test cannot count a NaN) is answered on the recorded logs, without a build: see "Finding F2".**

Branch `topic/b580-fused-port`, parent `51d9d757f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`.
Final configuration: the branch head with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1`.
Host `fedora` (the owner's desktop), Arc B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`. Artifacts `/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.
ANV Mesa 26.2.3 throughout; kernel 7.2.8-200.fc44 until the reboot of 2026-10-09 01:17 UTC, 7.2.9-200.fc44 after
it; GT frequency policy as found and unchanged (`min_freq` 1200, `max_freq` 2850, `rp0` 2850 MHz,
`power_profile` `[base] power_saving`).

## Running now

Nothing. No GPU job and no build was started after `tools/chain13.sh` ended `CHAIN13_DONE` at 11:06 UTC; the
closing work of 15:25 to 15:26 UTC read recorded files only (and compiled the two CPU binaries of
`sarc/tools/check.sh --no-build`).

## Closing (2026-10-09, after the owner decision of 15:25 UTC)

- Measured build: `topic7` = `e1e450530`; `git diff --name-only e1e450530 HEAD -- . ':(exclude)openspec/**'` is
  empty, so the branch head builds the binaries that were gated and timed. No local patch.
- Final verification on that build (R11.1): `s6-final` `GATE_PASS` (unmodified `verify.sh` the same as the parent
  snapshot, SDPA tiers `all` / `extended` / `full` 12 passes each with 0 mismatches, finite error and
  `pairing=ok`, next token SAME in 18 of 18 comparisons); shipped SPIR-V golden PASS (53 variants); reference
  error `raw/final-ref`, now also `results/b580/sdpa-error/final-ref-{full,extended,peaked}.csv`.
- Timed sessions (R11.2): `s6-final` against the parent, `s7-pristine` against the pristine parent `6a7cc8cc6`.
- `sarc/tools/check.sh --no-build`, run again at closing (`.artifacts/logs/check-close.out`, rc 0):

  ```
  == 1 zone rule vs origin/release/1.5
  == 2 twin wrappers
  == 3 test_sarc_select
  test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
  [sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
  test_sarc_select: PASS (1562 checks, 37 rows, 213 candidates, dev zone linked, unverified on)
  check.sh: PASS
  ```

  The release-zone edits on this branch are the two D4 hooks only (`fab9606c3`, D4.1; `0ffc84a2d`, D4.3; owner
  decision 2026-10-05); `Pipeline.cpp` is not edited.
- Final `b580-fused1` (`s6-final`) against the published numbers (`cells.csv`): +75.0 / +108.0 / +59.7 / +95.3 /
  +40.6 / +87.2 %, geomean +76.2 % (the published run of 8B 8da4w was disturbed: first campaign, baseline note).
  The parent of `s6-final` reads +0.64 / 0.00 / +0.25 / +0.31 / +0.11 / 0.00 % against the expected parent
  numbers of the task file.

## Finding F2 of the last review: the correctness test does not count a NaN (answered on the recorded logs)

An earlier version of the shared-memory reading below said that NaN rows written by the kernel's fallback are
something "every correctness tier would report". **That was wrong.** `test_llama_microbench.cpp` counts a
mismatch with `diff > thresh` (line 2119), which is false when `diff` is NaN, so a NaN output leaves
`mismatches=0` and the case prints PASSED. The comparator is the first campaigns' and was not changed here; the
tolerances are untouched.

What does see a NaN is the `[sdpa-error]` record the same test prints for every case: `rms_err` is the square
root of a sum over every output element (lines 2165 to 2177), so one NaN or infinity in the output or in the
reference makes it non-finite (`max_abs_err` alone would not: `std::max` drops a NaN). On that basis:

- `tools/gate_check.py` now requires, for every pass of every SDPA tier, one `[sdpa-error]` record per case
  with finite `rms_err`, `max_abs_err` and `ref_rms`; a non-finite or missing record fails the gate. No
  tolerance, threshold or file under `sarc/tools` changed; the requirement can only turn a pass into a fail.
- Regression test `tools/test_gate_finite.sh` (`results/b580/gate-recheck/test_gate_finite.out`,
  `TEST_GATE_FINITE_OK`; run again end to end after the second review round, 2026-10-09 15:55 UTC, with
  `.artifacts/tmp` writable: rc 0, output identical to the recorded file): on a copy of the `s6-final` gate files the gate passes; with `rms_err` set to `nan`,
  `-nan` or `inf`, with `ref_rms=nan`, or with one record removed, in one log whose four cases still read
  `mismatches=0 ... PASSED`, it fails on that pass.
- The five recorded gates decided again with the requirement (`results/b580/gate-recheck/`; the `gate.txt` of
  each session is kept as written): `s2-c1`, `s3b-c2`, `s4-final`, `s6-final` `GATE_PASS`, 37 PASS lines each;
  `s3-c2` `GATE_FAIL` with its 7 timing lines as before; the parent control `CONTROL_RECORDED`. All 960 records
  of the five gates (5 x 12 passes x 16 cases) are finite, so **no measured gate saw a NaN or infinite output**,
  and the fallback did not fire in any of them. The reference-error runs (`c1-ref6`, `c2-ref6-*`, `final-ref`)
  hold finite values in every row as well.
- Not done, and why: the test's own counter is not changed. That is a source change outside `openspec/`, after
  which the branch head would no longer be the measured build, and the owner decision of 15:25 UTC orders no
  rebuild, no re-gate and no new session. It belongs to the later change that repairs F1, which gates every
  device again: count `!(diff <= thresh)` or test `std::isfinite` on output and reference in the microbench.
  The production-diff and the `verify.sh` correctness cases are other comparators and were not audited for the
  same blind spot; they run the linear kernels, which have no such fallback.

## Sessions under the authorized protocol (chain 13, from 09:37 UTC; build `topic7` = `e1e450530`; no wait)

Every timed run started at once; `BUSYMAX` 5.0 %, `CLKMIN` 2635 MHz, 7 repeats. Tok/s, median of 7 valid runs per
arm, arms interleaved, recomputed from `stage/<session>/raw/runs.csv`. The desktop was idle (`IdleHint=yes`,
`logs/idle_wait.log`).

`s7-pristine` (09:38 to 09:58 UTC): pristine `6a7cc8cc6` with no profile against `topic7` with `b580-fused1`; 84
timed runs, 84 valid; foreign engine time 0.25 to 0.42 %:

| cell | pristine | `b580-fused1` | total gain | spread pristine / final |
|---|---:|---:|---:|---|
| 1B 4w | 8677.97 | 15058.80 | **+73.53 %** | 0.4 / 0.7 % |
| 1B 8da4w | 8865.80 | 18285.70 | **+106.25 %** | 0.4 / 0.9 % |
| 3B 4w | 3442.02 | 5461.33 | **+58.67 %** | 0.2 / 0.3 % |
| 3B 8da4w | 3524.96 | 6849.50 | **+94.31 %** | 0.2 / 0.3 % |
| 8B 4w | 1726.81 | 2384.17 | **+38.07 %** | 0.1 / 0.3 % |
| 8B 8da4w | 1845.05 | 3141.10 | **+70.24 %** | 0.4 / 0.3 % |

Geomean **+72.05 %** (`s5-pristine`, behind the wait: +71.77 %). The session ends `E2E5_INCOMPLETE
8b-8da4w:nexttoken_DIFFER_DIFFER_SAME`, as `s5-pristine` did: all 84 timed runs are valid and every cell has its
7; the incomplete mark is the next-token item of 8B 8da4w against the pristine parent, which the first campaign
recorded for `b580-refine3` and accepted under the reference-error rule. It is a timing statement, not a gate.
(The repeat rule of the owner decision of 09:55 UTC is for a cell short of valid runs; no cell is.)

`s3c-c2` (10:00 to 10:06 UTC): `topic7` with `b580-fused1` against `topic7` with `b580-fused2`, timed only; 84
timed runs, 84 valid; foreign engine time 0.24 to 0.43 %; `E2E5_OK`, next token SAME in all 18 comparisons:

| cell | candidate 1 | candidate 2 | gain |
|---|---:|---:|---:|
| 1B 4w | 14948.90 | 15058.80 | +0.74 % |
| 1B 8da4w | 18285.70 | 18285.70 | +0.00 % |
| 3B 4w | 5461.33 | 5446.81 | -0.27 % |
| 3B 8da4w | 6849.50 | 6849.50 | +0.00 % |
| 8B 4w | 2373.12 | 2373.12 | +0.00 % |
| 8B 8da4w | 3141.10 | 3141.10 | +0.00 % |

Geomean **+0.08 %**, every cell inside +-2 % (`s3b-c2`, behind the wait: -0.13 %). Candidate 2 is not a gain by
the `noise_band` rule; the final stack stays `b580-fused1`. Its gate items that do not depend on the start time
are those of `s3b-c2` (36 PASS lines).

`s6-final` (10:06 to 11:06 UTC; timed runs 10:46 to 11:04): `parent2` with the parent environment against
`topic7` with `b580-fused1`, the full gate (`gate_sdpa.sh`). The runner had the `busy_unreadable` predicate of
the owner decision of 09:55 UTC from before its timed runs. 84 timed runs, 84 valid; foreign engine time 0.26 to
0.46 %:

| cell | parent `b580-refine3` | final `b580-fused1` | gain | spread parent / final | next token (2048 / real-text / 1792-token prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 13044.60 | 15058.80 | **+15.44 %** | 0.6 / 0.7 % | SAME / SAME / SAME |
| 1B 8da4w | 15170.40 | 18285.70 | **+20.54 %** | 0.7 / 0.9 % | SAME / SAME / SAME |
| 3B 4w | 5197.97 | 5461.33 | **+5.07 %** | 0.5 / 0.3 % | SAME / SAME / SAME |
| 3B 8da4w | 6420.06 | 6849.50 | **+6.69 %** | 0.3 / 0.3 % | SAME / SAME / SAME |
| 8B 4w | 2308.91 | 2381.40 | **+3.14 %** | 0.3 / 0.1 % | SAME / SAME / SAME |
| 8B 8da4w | 2998.54 | 3145.93 | **+4.92 %** | 0.1 / 0.2 % | SAME / SAME / SAME |

Geomean **+9.12 %**, every cell outside the +-2 % band (`s4-final`, behind the wait: +8.82 %; `s2-c1`, desktop in
use: +9.18 %). `stage/s6-final/gate.txt`: `GATE_PASS`, 37 PASS lines (the 36 of the earlier gates and the new
line "no run stored valid without a readable foreign engine share"), no FAIL line: SDPA tiers `all` /
`extended` / `full` 12 passes each with 0 mismatches and `pairing=ok`, unmodified `verify.sh` with the final
environment the same as the parent snapshot, next token SAME in all 18 comparisons. A plain pass on the timing
and correctness items, on pipelines created without the full-subgroups flag (F1; known defect, owner decision
of 15:25 UTC). Reference error, the extra tiers, the golden check and the roofs do not depend on the start time
and are those of the first closing on the same build (below).

Chain 13 ended `CHAIN13_DONE` at 11:06 UTC. Nothing is running.


## Result of the first closing (sessions `s4-final` and `s5-pristine`, 07:28 to 08:49 UTC, build `topic7` = `e1e450530`); superseded as evidence by chain 13

**These two sessions were started through `busy_wait` (it read 0.3 to 0.4 % and let them start at once; the
delay was the 5 s reading). They are kept as measured and are no longer the evidence of the close: `s6-final`
and `s7-pristine` replace them. The pipelines measured here are the ones finding F1 is about.**

Build `topic7` is the export of the committed head `e1e450530`, no local patch (later commits change files
under `openspec/` only; its sources outside `openspec/` are also those of `topic6` = `247d08851`, the build
candidate 1 was first gated on). Tok/s, median of 7 valid runs per arm, arms interleaved, recomputed from
`stage/<session>/raw/runs.csv`. Both sessions: 84 timed runs, 84 valid; foreign engine time 0.31 to 0.33 % per
cell (median), 0.44 % at most; the desktop was idle (the owner had left it), arm spreads 0.1 to 0.9 % against
the parent and 0.1 to 1.8 % against the pristine parent.

| cell | parent `b580-refine3` | final `b580-fused1` | gain | pristine `6a7cc8cc6`, no profile | final (that session) | total gain | published (`cells.csv`) | final vs published |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | 13044.60 | 14948.90 | **+14.60 %** | 8677.97 | 14948.90 | **+72.26 %** | 8605.04 | +73.7 % |
| 1B 8da4w | 15283.60 | 18285.70 | **+19.64 %** | 8865.80 | 18285.70 | **+106.25 %** | 8789.70 | +108.0 % |
| 3B 4w | 5197.97 | 5446.81 | **+4.79 %** | 3442.02 | 5461.33 | **+58.67 %** | 3419.03 | +59.3 % |
| 3B 8da4w | 6420.06 | 6849.50 | **+6.69 %** | 3524.96 | 6826.67 | **+93.67 %** | 3506.85 | +95.3 % |
| 8B 4w | 2298.54 | 2373.12 | **+3.24 %** | 1726.81 | 2384.17 | **+38.07 %** | 1693.96 | +40.1 % |
| 8B 8da4w | 2998.54 | 3145.93 | **+4.92 %** | 1843.38 | 3141.10 | **+70.40 %** | 1680.07 | +87.2 % (the published run of this cell was disturbed: first campaign, baseline note) |

Geomean **+8.82 %** over the parent, every cell outside the +-2 % band; **+71.77 %** over the pristine parent;
+75.8 % over the published numbers. The parent reads +0.27 % (geomean; -0.34 to +0.75 % per cell) against the
first campaign's `s6-final`, on the new kernel 7.2.9, so the reboot did not move the baseline. The session of
candidate 1 on the desktop in use (`s2-c1`, build `topic6`) had read +9.18 % (per cell +17.1 / +19.0 / +5.4 /
+6.8 / +3.5 / +4.3 %): the two sessions agree within 2.5 points in 1B 4w and within 0.7 points elsewhere; the
closing session is the quiet one and is the number to quote.

Final verification, all on `topic7` with `b580-fused1` (`stage/s4-final/gate.txt`: `GATE_PASS`, 36 PASS lines, no
FAIL line):

- SDPA correctness, 12 passes each: tiers `all` / `extended` / `full` (gate items) and `peaked` / `fused`
  (reported): 0 mismatches in every case run, `pairing=ok`, the fused kernel the only attention kernel.
- Unmodified `verify.sh` with the final environment against the parent snapshot `s0-parent-verify`, line by
  line with the rates removed: the same. Next token parent vs final: SAME in all six cells on the three
  prompts, so the result is a plain pass; no item needs decision D1 or D3.
- Error against the fp32 reference (`.artifacts/raw/final-ref/{full,extended,peaked}.csv` and its logs; copied to
  `results/b580/sdpa-error/final-ref-*.csv` at closing): not larger than
  the parent's in all 4 `full` and all 8 `extended` cases (rms 2.02e-5 to 2.05e-5 against 2.76e-5 to 2.79e-5 at
  S = 2048); in 1 of 5 synthetic `peaked` cases the maximum error is larger (2.50e-3 against 2.32e-3) with a
  smaller rms; the same values as on `topic6`.
- Shipped SPIR-V golden: PASS, 53 variants, on `topic7`, `pristine` and `parent2`.
- Hook condition (D4): `verify.sh` with no environment on `topic7` against the parent's: `VERIFY_SAME`, 34
  lines; `sarc/tools/check.sh --no-build`: `check.sh: PASS` (`test_sarc_select` 1240 checks / 31 rows without
  the dev zone, 1562 checks / 37 rows / 213 candidates with it), `.artifacts/logs/check-final.out`.
- Against the pristine parent the next token differs in 8B 8da4w on the 2048-token and the real-text prompt
  (SAME in the other 16 comparisons): that is the item the first campaign recorded for `b580-refine3` itself and
  accepted under the reference-error rule; this campaign's stack does not move any token against its parent.
  `e2e5.sh` therefore ends `s5-pristine` with `E2E5_INCOMPLETE 8b-8da4w:nexttoken_DIFFER_DIFFER_SAME`; the
  session is a timing statement, not a gate.
- Decode (`s4-final`, 32 tokens, medians of 5): final / parent 0.98462 / 0.99148 / 0.99150 / 0.98960 / 0.99830 /
  0.99603: inside the band, but below 1 in all six cells, here and in `s2-c1` (0.99177 to 0.99918). Decode does
  not run the fused kernel; the cost was not located.

Where the gain comes from (warm ETDump, ms per 2048-token prefill, attention kernels by name,
`stage/s4-final/trace/attention.csv`):

| cell | arm | dispatch total | QK^T | softmax | attn*V | fused kernel | K/V copy pass | attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 156.4 | 7.0 | 17.3 | 8.0 | | | 32.3 |
| 1B 4w | final | 135.3 | | | | 10.8 | 0.3 | 11.0 |
| 1B 8da4w | parent | 134.1 | 6.9 | 17.2 | 7.9 | | | 32.1 |
| 1B 8da4w | final | 112.6 | | | | 10.7 | 0.3 | 10.9 |
| 3B 4w | parent | 391.8 | 13.3 | 22.8 | 14.6 | | | 50.7 |
| 3B 4w | final | 373.9 | | | | 30.5 | 0.9 | 31.4 |
| 3B 8da4w | parent | 319.1 | 13.0 | 22.7 | 14.5 | | | 50.2 |
| 3B 8da4w | final | 299.8 | | | | 29.9 | 0.8 | 30.8 |
| 8B 4w | parent | 883.3 | 20.0 | 34.8 | 21.7 | | | 76.5 |
| 8B 4w | final | 857.1 | | | | 46.6 | 0.9 | 47.5 |
| 8B 8da4w | parent | 680.7 | 19.4 | 34.4 | 21.1 | | | 74.8 |
| 8B 8da4w | final | 651.3 | | | | 44.4 | 0.9 | 45.2 |

Attention falls by 66 % on 1B and by 38 to 40 % on 3B and 8B; everything else moves by -0.3 to +2.8 ms. The
whole gain is the attention family, and most of what was removed is the softmax's traffic over the S x S
matrix.

Roofs, re-measured (igpu-roofline plan `fast`, `roofline/final`, 2026-10-09 08:49 to 09:12 UTC, Mesa 26.2.3,
runner `810e098c8abb`, clocks not pinned; `results/b580/roofline/final/REPORT.md`; the report step was run with
the first campaign's Python environment, which has matplotlib): fp16 matrix 112.86 TFLOP/s, fp16 -> fp32 matrix
115.68 TFLOP/s, int8 matrix 231.36 TOP/s, global read 465 GB/s (first campaign: 112.9 / 231.4). Linear kernels
in the final stack, time-weighted over the prefill GEMMs of the warm ETDump (`tools/roof_util.py`): 4w 45.0 /
44.2 / 42.8 TFLOP/s = **39.9 / 39.2 / 38.0 %** of the fp16 matrix roof (1B / 3B / 8B); 8da4w 69.3 / 69.1 / 65.9
TOP/s = **30.0 / 29.9 / 28.5 %** of the int8 matrix roof; unchanged from the parent, as expected. The fused
kernel, counted as if every one of the S x S scores were computed (2 x 2 x S^2 x head_dim x heads per layer over
the kernel time of screen 5): 47 / 45 / 47 TFLOP/s = 41 / 39 / 41 % of the fp16 -> fp32 roof; about half the
blocks are behind the causal mask and are skipped, so the rate on the work actually done is roughly half of
that. This attention figure is an estimate from the kernel times, not a measured FLOP count.

What limits further progress: after this port attention is 8 to 10 % of the prefill's dispatch time on 1B and 3B and
5.5 to 7 % on 8B; the linear kernels are 51 % (1B 8da4w) to 78 % (8B 4w) of it, at 38 to 40 % (4w) and 28 to
30 % (8da4w) of their roofs. The fused kernel itself is bounded on this card by the register file (4096 bytes of 16-lane values per
thread), which is why a workgroup had to be split into 4 or 8 subgroups that exchange a block's scores through
shared memory; a wider register file or a 16 x 16 x 16 matrix shape would allow the 780M's single-subgroup
form. The next gain is in the linear kernels, not in attention.

Negative results, kept with their numbers below: the straight port of the 780M's shapes (0.14x to 0.35x of the
parent's three kernels); accumulators in shared memory (slower in every case); one column of score tiles live
(no effect); candidate 2, the fp32 no-tail softmax for the calls the fused kernel does not take (-0.13 %);
candidate 2's first session without a single valid run on the desktop in use.

## Candidate 2, `b580-fused2`: GATE_PASS, -0.13 % geomean, not adopted (session `s3b-c2`, 05:04 to 06:15 UTC)

Parent of this comparison = candidate 1: build `topic7` with `b580-fused1`; candidate = the same build with
`b580-fused2`. Tok/s, median of 7 valid runs per arm, arms interleaved; recomputed from
`stage/s3b-c2/raw/runs.csv`:

| cell | candidate 1 | candidate 2 | gain | spread c1 / c2 | foreign engine time, median / max | next token (2048 / real-text / 1792-token prompt) |
|---|---:|---:|---:|---|---|---|
| 1B 4w | 14733.80 | 14733.80 | +0.00 % | 0.7 / 2.1 % | 1.48 / 1.93 % | SAME / SAME / SAME |
| 1B 8da4w | 17964.90 | 17808.70 | -0.87 % | 1.8 / 0.0 % | 1.52 / 2.08 % | SAME / SAME / SAME |
| 3B 4w | 5361.26 | 5375.33 | +0.26 % | 2.1 / 0.5 % | 1.44 / 1.57 % | SAME / SAME / SAME |
| 3B 8da4w | 6736.84 | 6714.75 | -0.33 % | 1.0 / 2.9 % | 1.42 / 1.53 % | SAME / SAME / SAME |
| 8B 4w | 2329.92 | 2327.27 | -0.11 % | 0.3 / 0.5 % | 1.50 / 1.91 % | SAME / SAME / SAME |
| 8B 8da4w | 2994.15 | 3002.93 | +0.29 % | 5.6 / 5.1 % | 3.26 / 4.99 % | SAME / SAME / SAME |

Geomean **-0.13 %**, every cell inside +-2 %. 86 timed runs, 85 valid, 1 rejected and replaced (8B 8da4w parent
r2, foreign engine time 5.94 %). The replacement pair left the candidate arm of 8B 8da4w with 8 valid runs;
`summarize.py` takes the first 7 (3002.93, +0.29 %); over all 8 the median is 3014.02 (+0.66 %) and the geomean
-0.07 %. The 1B times are quantised by the runner's 1 ms timer (139 ms: one step is 0.7 %). Desktop in use
(`IdleHint=no`). **The session started behind `busy_wait`, one minute later than the authorized protocol would
have started it (first reading 5.4 %, then 2.4 %; `logs/busy_wait.log`); its timing is kept and is repeated as
`s3c-c2` without the wait.**

Gate (`stage/s3b-c2/gate.txt`): `GATE_PASS`, 36 PASS lines. On the timed prompt the two arms dispatch the same
kernels (the fused kernel takes every attention call), so no gain was possible there; the softmax variant acts
on the calls the fused kernel does not take (the 1972-token prompt, decode). Logits probe of the first session
(`s3-c2`, same sources): candidate 2 bit-identical to candidate 1 in all 35 aligned real-text windows of all six
cells. Reference error (`c2-ref6`, `topic6`), the softmax variant between the parent's QK^T and attn*V kernels
(`b580-refine3-nzf`), rms / maximum, parent -> variant: 1B heads 2.76e-5 / 1.19e-3 -> 2.10e-5 / 9.1e-4, 3B
2.79e-5 / 1.15e-3 -> 2.07e-5 / 7.8e-4, 8B 2.76e-5 / 1.06e-3 -> 2.06e-5 / 8.9e-4, S = 1024 at input_pos 1024
1.29e-5 / 1.24e-4 -> 9.0e-6 / 6.9e-5: not larger in all 4 `full`, all 8 `extended` and all 5 `peaked` cases.
`b580-fused2` itself: as candidate 1 (the same one `peaked` case with a larger maximum). Decode (`s3-c2`,
medians of 5): 0.988 to 1.015 of candidate 1.

By the `noise_band` rule of `thresholds.txt` a difference inside +-2 % is not a gain, so **the final stack is
`b580-fused1`**. (The actor had also written a `final_stack` rule into `thresholds.txt` at 03:17 UTC; it is
removed, `thresholds-history.md`, and is not needed for this.)
Candidate 2 stays selectable as profile `b580-fused2`; what it offers is a smaller error on the calls the fused
kernel does not take, not speed.

Stop rule: "after candidate 2 whatever its result" holds. (Candidate 1 gained more than 2 %, so the
two-consecutive rule is not what ends the campaign.)

## Candidate 2, first session `s3-c2` (01:58 to 03:15 UTC): gate failed on timing only; the desktop held a third of the card

`b580-fused2` (candidate 1 + the fp32 no-tail softmax `4070ti_nzf` for the calls the fused kernel does not take)
against candidate 1, both build `topic6`. `stage/s3-c2/gate.txt`: 29 PASS lines (SDPA tiers 12 passes each with 0
mismatches, unmodified `verify.sh` against the parent snapshot, next token, traces) and 7 FAIL lines, all of them
"7 valid runs per build" and "session complete". `stage/s3-c2/raw/runs.csv`: 252 rows, 228 timed runs (19 per
arm and cell: 7 + 12 extra pairs). 227 of them are stored invalid with reason `foreign_busy`: the desktop's share
of the engine time in the prefill window was 32.87 to 45.15 %, limit 5.0 %; the top foreign client in every run
was pid 243998, a Discord renderer (the desktop session also had firefox, slack and gnome-remote-desktop open).
The remaining one, `prefill-3b-8da4w-parent-r12.log`, is stored `valid=1` with an empty reason although its
share reads -5274.35 % (a client's cycle counter went backwards between two samples): not a reading. By the
keyed adjudication `tools/adjudication.csv` it is not countable (`busy_unreadable`); **countable timed runs: 0 of
228** (stored valid: 1). `runs.csv` and its committed copy are kept as written, the gate stays `GATE_FAIL`, and
`summarize.py` and `gate_check.py` count through `tools/adjudicate.py`, so that neither this row nor any later
row without a share between 0 and 100 % can be counted (`thresholds-history.md`). No median, no gain:
**candidate 2 has no timing result from this session.** It is a session without a measurement, kept where it
is as the record. (An earlier version of this section said 237 rows and 216 timed runs: that count was taken
while the session was still writing.)

What the actor then did wrong, and what was undone (reviewer round, commit `917b471af`): it added a wait on the
desktop's load before timed sessions (`busy_wait`), a validity test in `e2e5.sh` and two entries in
`thresholds.txt`. The owner decision of 00:22 UTC orders every timed run to start at once, and `thresholds.txt`
is not changed after a candidate's numbers exist. All of it is removed; the history, and what the wait did to
which session, is in `thresholds-history.md`.

Smoke runs before the gate (`raw/c2-smoke/`): `b580-refine3-nzf` tiers `all` / `extended` / `full` / `peaked` 4 /
8 / 4 / 5 cases with 0 mismatches, softmax `sarc_sdpa_attn_weights_softmax_buffer_half_4070ti_nzf`; the control
with the cooperative-matrix kernels disabled under `b580-fused2`: 8 of 8 cases with 0 mismatches (the test
prints FAILED there by design: it requires the cooperative-matrix kernels).

From 01:17 UTC on the kernel is 7.2.9-200.fc44 (7.2.8-200 before): the baseline and A/A (`s1-aa2`), the screens
and session `s2-c1` were measured on 7.2.8; `s3-c2` and the closing sessions run on 7.2.9. Every session
compares two arms interleaved on one kernel; the closing session against the parent shows whether the parent's
absolute numbers moved with it.

### Candidate 1: error against the fp32 reference (`c1-ref6`, staged binary of `s2-c1`, 01:02 UTC)

`.artifacts/raw/c1-ref6/{full,extended,peaked}.csv` (copied to `results/` by the collection); rms / maximum
absolute error, parent's kernels -> `b580-fused1`, 0 mismatches in every case of both arms:

| tier | case | parent | `b580-fused1` | not larger (rms / max) |
|---|---|---|---|---|
| `full` | 1B heads, S = 2048 | 2.76e-5 / 1.19e-3 | 2.05e-5 / 7.2e-4 | yes / yes |
| `full` | 3B heads, S = 2048 | 2.79e-5 / 1.15e-3 | 2.05e-5 / 7.1e-4 | yes / yes |
| `full` | 8B heads, S = 2048 | 2.76e-5 / 1.06e-3 | 2.02e-5 / 7.9e-4 | yes / yes |
| `full` | 8B heads, S = 1024 at input_pos 1024 | 1.29e-5 / 1.24e-4 | 8.96e-6 / 6.9e-5 | yes / yes |
| `extended` | 8 cases | | | yes / yes in all 8 |
| `peaked` | `peaked_tiny_gqa_s256` | 4.06e-4 / 2.32e-3 | 3.77e-4 / **2.50e-3** | yes / **no** |
| `peaked` | the other 4 cases | | | yes / yes |

The gate criterion of decision D3 is the production-shape cases (`full`): met (`SDPA_ERROR_OK`), and it is not
needed for acceptance, because no next token moved. The `peaked` tier is synthetic (sharp rows, to exercise the
rescale of the one-pass form) and is reported, not a gate item (`chain8.sh`, fixed before the run): in one of
its five cases the candidate's maximum error is 8 % above the parent's while its rms error is 7 % below.

## Candidate 1, `b580-fused1`, first gate: GATE_PASS, +9.18 % geomean on the desktop in use (session `s2-c1`, build `topic6`)

Session `s2-c1` (2026-10-09 00:34 to 00:50 UTC): parent = build `parent2` (`51d9d757f`) with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`; candidate = build `topic6` (`247d08851`) with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1`. Tok/s, median of 7 valid runs per arm (the
repeat count the A/A asked for), arms interleaved; recomputed from `stage/s2-c1/raw/runs.csv`:

| cell | parent | candidate 1 | gain | spread parent / cand | foreign engine time, median / max | next token (2048 / real-text / 1792-token prompt) |
|---|---:|---:|---:|---|---|---|
| 1B 4w | 12487.80 | 14628.60 | **+17.14 %** | 3.7 / 4.8 % | 2.15 / 3.69 % | SAME / SAME / SAME |
| 1B 8da4w | 14840.60 | 17655.20 | **+18.97 %** | 5.0 / 2.6 % | 1.65 / 3.36 % | SAME / SAME / SAME |
| 3B 4w | 5019.61 | 5291.99 | **+5.43 %** | 0.7 / 2.8 % | 2.31 / 2.74 % | SAME / SAME / SAME |
| 3B 8da4w | 6206.06 | 6627.83 | **+6.80 %** | 1.8 / 1.9 % | 2.21 / 2.55 % | SAME / SAME / SAME |
| 8B 4w | 2228.51 | 2306.31 | **+3.49 %** | 0.7 / 2.7 % | 2.24 / 3.55 % | SAME / SAME / SAME |
| 8B 8da4w | 2904.96 | 3029.59 | **+4.29 %** | 0.9 / 2.1 % | 2.29 / 2.91 % | SAME / SAME / SAME |

Geomean **+9.18 %**, every cell outside the +-2 % band (A/A +0.26 %). 84 timed runs, none rejected. **The
desktop was in use for the whole session** (seat0 `IdleHint=no` at every one of its runs,
`logs/idle_wait.log`; the owner lifted the idle wait at 00:22 UTC): foreign engine time 1.7 to 2.3 % per cell
(median), 3.7 % at most, under the 5 % limit; both arms read 2 to 3 % lower than on the idle desktop
(`s1-aa2`), and the arm spreads are 0.7 to 5.0 %. The gains are a ratio of two arms interleaved under the same
conditions; the absolute tok/s are not idle-desktop numbers. The closing session repeats the measurement.

Gate (`gate_sdpa.sh`, `stage/s2-c1/gate.txt`): `GATE_PASS`, 36 PASS lines, no FAIL line. SDPA correctness 12
passes x tiers `all` / `extended` / `full` (4 / 8 / 4 cases), 0 mismatches and `pairing=ok` in all 192 case
runs with the fused kernel the only attention kernel dispatched. Unmodified `verify.sh` with the candidate
environment: identical to the parent snapshot `s0-parent-verify` line by line with the rates removed
(`tools/verify_diff.py`: `VERIFY_SAME`, 34 lines): 12 of 12 production-diff cases ALL PASSED, 28 of 28 numeric
and 4 of 4 rank-3 correctness cases, default vs tiled SAME on both prompts for both schemes, decode 31 tokens.
Next token parent vs candidate SAME in all six cells on all three prompts, so no next-token item needs the
near-tie or the reference-error decision; the candidate does change the arithmetic, and its error against the
fp32 reference is in the table above (`c1-ref6`).

Where the gain comes from (warm ETDump of both arms, ms per 2048-token prefill, attention kernels by name,
`stage/s2-c1/trace/attention.csv`; everything else in `trace/report/evidence/trace/families.csv`):

| cell | arm | dispatch total | QK^T | softmax | attn*V | fused kernel | K/V copy pass | attention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 160.9 | 7.0 | 17.5 | 7.9 | | | 32.4 |
| 1B 4w | candidate 1 | 141.8 | | | | 11.3 | 0.2 | 11.6 |
| 1B 8da4w | parent | 137.1 | 7.1 | 17.6 | 8.3 | | | 33.0 |
| 1B 8da4w | candidate 1 | 117.7 | | | | 10.9 | 0.2 | 11.2 |
| 3B 4w | parent | 407.9 | 13.8 | 25.2 | 15.4 | | | 54.5 |
| 3B 4w | candidate 1 | 388.8 | | | | 31.6 | 0.9 | 32.5 |
| 3B 8da4w | parent | 334.3 | 13.0 | 22.8 | 14.6 | | | 50.4 |
| 3B 8da4w | candidate 1 | 307.7 | | | | 30.4 | 0.8 | 31.2 |
| 8B 4w | parent | 916.8 | 20.3 | 36.9 | 22.1 | | | 79.3 |
| 8B 4w | candidate 1 | 883.8 | | | | 48.0 | 1.0 | 48.9 |
| 8B 8da4w | parent | 704.3 | 20.0 | 35.5 | 21.7 | | | 77.1 |
| 8B 8da4w | candidate 1 | 680.3 | | | | 44.7 | 1.0 | 45.7 |

The attention family goes from 32.4 to 11.6 ms on 1B (-64 %), 54.5 to 32.5 ms on 3B (-40 %) and 79.3 to 48.9 ms
on 8B (-38 %); the other families move by less than 1 % of the prefill. What the fused kernel removes is mostly
the softmax's traffic over the S x S matrix (17.5 of 32.4 ms on 1B), which was larger than QK^T and attn*V
together.

Kernel level, for predicting the B70 (L7; screen 5, idle desktop, us per layer at S = 2048): the three kernels
1989 (1B) / 1771 (3B) / 2297 (8B); the fused kernel with its copy pass 724 / 1139 / 1464: 2.75x / 1.55x / 1.57x.

## Candidate 1: `b580-fused1`, selected by screen 5 (`results/b580/screens/screen5-select.csv`, build `topic5`)

Three rounds on the idle desktop, cooled before every run; kernel time per layer at S = 2048, us (fused kernel
+ copy pass; for the parent QK^T + softmax + attn*V); each round's value:

| head_dim | variant | 1B | 3B | 8B | vs parent's three kernels |
|---|---|---|---|---|---|
| | parent `b580-refine3` | 1989 / 1990 / 1985 | 1766 / 1771 / 1877 | 2297 / 2454 / 2295 | 1.00x |
| 64 | incumbent `d64_t32x32s32m8ro` (the 780M's) | 5939 / 5970 / 5737 | | | 0.34x |
| 64 | **`d64_t16x64s16m8g4roj`** | **724 / 724 / 725** | | | **2.75x** |
| 64 | `d64_t16x64s16m8g4oj` | 821 / 821 / 819 | | | 2.42x |
| 128 | incumbent `d128_t16x64s32m8ro` (the 780M's) | | 11004 / 10635 / 11077 | 14419 / 14357 / 14688 | 0.16x |
| 128 | **`d128_t16x128s16m8g8oj`** | | **1139 / 1139 / 1191** | **1464 / 1465 / 1462** | **1.55x / 1.57x** |
| 128 | `d128_t16x64s16m8g4oj` | | 1230 / 1209 / 1204 | 1563 / 1561 / 1505 | 1.47x |

By the rule fixed in `thresholds.txt` (at least 3 % below the incumbent in every round for every model of the
head_dim; among those the lowest median) the challengers win by a factor of 8 to 10, and `b580-fused1` is
`d64_t16x64s16m8g4roj` + `d128_t16x128s16m8g8oj`. The challengers were the two fastest variants per head_dim of
the one-round look `screen4-fused` (11 profiles, `results/b580/screens/screen4-fused.csv`); 47 variants exist
in the yaml, all of them what was built on the way, none found by a search. For the B70 (same driver, same
matrix shapes, same register file): expect the same two variants.

What the pair is: 16 query rows per workgroup. Head_dim 64: 4 subgroups (64 lanes), 64-column blocks, each
subgroup owns one score-tile column and one 16-wide slice of head_dim (2 accumulator tiles), Q tiles in
registers. Head_dim 128: 8 subgroups (128 lanes), 128-column blocks, each subgroup owns one score-tile column
and one 16-wide slice of head_dim, Q tiles loaded per product. One pass (running row maximum). It serves a
prefill call when S and `input_pos` are multiples of 64 (head_dim 64) or 128 (head_dim 128); the 2048-token
prompt and the 1792-token prompt `r1304.txt` qualify, the 1972-token `prompt_check.txt` and decode do not and
run the parent's kernels.

## Why the straight port is slow here, and the form that is not (22:10 to 22:55 UTC)

Compiler statistics of the fused pipelines (`INTEL_DEBUG=cs` with the shader cache disabled: a compile-time
dump of the test process, no hardware counter; `results/b580/compile/topic4-fused-kernels.txt`). ANV compiles
the kernel for 16 lanes with 128 registers:

| variant | instructions | spills : fills | kernel us per layer (where measured) |
|---|---:|---:|---|
| `d64_t8x32 ro` (the only form of screen 1 that beat the three kernels) | 1304 | 32 : 57 | 1375 (1B), idle |
| `d64_t16x32 ro` | 3974 | 204 : 313 | 2585 to 2718 |
| `d128_t8x64 ro` | 4178 to 5187 | 199 : 336 to 268 : 470 | 3520 (8B) |
| `d128_t16x64 ro` (the 780M's shape at 16 lanes) | 15093 to 16537 | 1114 : 1309 to 1250 : 1467 | 17056 (8B) |
| `d128_t8x64 g2 oj` (2 subgroups) | 1107 | 0 : 0 | |
| `d128_t8x64 g4 roj` (4 subgroups) | 768 | 0 : 0 | |
| `d128_t16x64 g4 oj` | 1155 | 0 : 0 | |
| `d64_t16x64 g4 roj` | 924 | 0 : 0 | |

A thread's register file holds 4096 bytes of 16-lane values; the fp32 accumulators of 8 rows x head_dim 128
alone are 4096 bytes. So one subgroup cannot own whole rows of head_dim 128 on this card, whatever the tile.

What was tried, in order (all correct in a pass of tiers `extended`, `peaked`, `fused`; `raw/c1-smoke2..3/`):

1. `j`: one column of score tiles live instead of a block's. No real effect (1B `t8x32`: 1311 against 1339 us,
   idle).
2. `a`: accumulators in shared memory, loaded and stored around each block. Slower in every case (one-shot look
   with the desktop in use, `results/b580/looks/look1-*.txt`: head_dim 128 5.7 to 11 ms against 3.5 ms). Shared
   memory is not a cheap extension of the register file here.
3. `g<G>`: **a workgroup of G subgroups.** Each subgroup owns 1 / G of the score-tile columns of a block and
   1 / G of the head_dim tiles of the accumulators; the block's scores and e values go through shared memory
   (`Psh`), which every subgroup reads, so the barriers become workgroup barriers and the rescale decision of
   the one-pass form is taken for the whole workgroup (an atomic flag in shared memory instead of
   `subgroupAny`). K and V are still read straight from the packed copies; no product changes and every tile
   accumulates in the same order; the row sum is added up per lane segment first, as before, with 4 or 8
   segments a row instead of 1 or 2. One-shot look with the desktop in use (`results/b580/looks/look2-*.txt`; the parent's kernels read
   2308 / 1788 / 2255 us for 8B / 3B / 1B in the same look, 0 to 9 % above their idle values):
   `d128_t16x64 g4 oj` 1508 / 1195 us (8B / 3B), `d128_t8x64 g4 roj` 2063 / 1716, `d128_t8x64 g2 oj` 2217 / 1732;
   `d64_t16x64 g4 roj` 726 us (1B), `d64_t8x32 g2 roj` 944, `d64_t8x32 oj` 1178. These are single disturbed runs:
   a direction, not a result.

## Calibration and baseline (session `s1-aa2`, 21:27 to 21:37 UTC, idle desktop)

Parent `parent2` (`51d9d757f`) against `topic1`, both with the parent environment; median of 5 valid runs per
arm, arms interleaved, 60 timed runs, none rejected; tok/s:

| cell | parent | topic, same environment | A/A | expected (`s6-final`) | parent vs expected | foreign engine time, median / max |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 12800.00 | 13044.60 | +1.91 % | 12962.00 | -1.25 % | 0.02 / 1.63 % |
| 1B 8da4w | 15283.60 | 15283.60 | 0.00 % | 15170.40 | +0.75 % | 0.01 / 1.74 % |
| 3B 4w | 5132.83 | 5145.73 | +0.25 % | 5184.81 | -1.00 % | 0.59 / 0.74 % |
| 3B 8da4w | 6360.25 | 6340.56 | -0.31 % | 6400.00 | -0.62 % | 0.73 / 0.92 % |
| 8B 4w | 2288.27 | 2288.27 | 0.00 % | 2306.31 | -0.78 % | 0.60 / 0.75 % |
| 8B 8da4w | 2976.74 | 2968.12 | -0.29 % | 2998.54 | -0.73 % | 0.64 / 0.94 % |

A/A geomean +0.26 %, every cell inside +-2 %; baseline within 1.3 % of the first campaign in every cell (limit
3 %); next token SAME in all six cells on the three prompts. Calibration (`tools/thresholds.txt`, dated block):
`CLKMIN` 2635 MHz, `BUSYMAX` 5.0 % (the floor), idle 42 C, and **7 repeats** for every later session, because the
1B 4w A/A is further than 1 % from 1: three of its five parent runs had 0.7 to 1.6 % foreign engine time
(firefox, ghostty) and read 1.3 to 2.5 % lower. The desktop was idle but not as quiet as in the first campaign
(0.00 % there).

## Screen 1, round 1: the 780M's shapes are slow here (`results/b580/screens/screen1-fused.csv`, build `topic1`)

Kernel time per layer at S = 2048, microbench `--sdpa`, us; one round on the idle desktop, cooled before every
run (round 2 had only begun when the screen was stopped; its two runs are in the run table). The parent's three
kernels: 8B 2400 (QK^T 571, softmax 1127, attn*V 702), 3B 1766, 1B 2069. Fused kernel + copy pass:

| head_dim 64 variant (1B) | us | vs parent | | head_dim 128 variant | 8B us | 3B us | vs parent |
|---|---:|---:|---|---|---:|---:|---:|
| `t8x32 ro` | 1375 | 1.50x | | `t8x64 ro` | 3520 | 2830 | 0.68x / 0.62x |
| `t8x64 ro` | 1511 | 1.37x | | `t8x64 o` | 3782 | 2853 | 0.63x |
| `t16x32 r` (two-pass) | 1721 | 1.20x | | `t8x64 r` (two-pass) | 4105 | 3467 | 0.58x |
| `t16x32 o` | 2358 | 0.88x | | `t8x32 ro` | 4208 | 3652 | 0.57x |
| `t16x32 ro` (= `b580-fused1` so far) | 2585 to 2718 | 0.78x | | `t16x64 o` | 7589 | 6193 | 0.32x |
| `t16x64 ro` | 2722 | 0.76x | | `t16x64 r` (two-pass) | 8150 | 6677 | 0.29x |
| `t32x32 s32 ro` (the 780M's) | 5982 | 0.35x | | `t16x32 ro` | 8307 | 7159 | 0.29x |
| | | | | `t16x64 s32 ro` (the 780M's) | 14643 | 10762 | 0.16x |
| | | | | `t16x64 ro` (= `b580-fused1` so far) | 17056 to 17170 | 12936 to 13929 | 0.14x |

Reading (an inference from these timings, not yet from a shader dump): the time follows the number of matrix
tiles a thread keeps live. With 16 lanes an 8 x 16 fp32 tile is 512 bytes and an fp16 tile 256; head_dim 128 at
8 rows keeps 8 accumulator, 8 Q and 4 score tiles = 8192 bytes, and 16 rows twice that; the only variants that
beat the three kernels keep 4096 bytes or less (head_dim 64 at 8 rows). The 780M's kernel assumed tiles stay in
registers. The new variants keep one column of score tiles live (`j`) and move the accumulators to shared
memory (`a`); neither changes a product or the order of a sum.

## The desktop went into use during the first A/A (20:50 UTC)

Session `s1-aa` (parent `parent2` against `topic1`, both with the parent environment) started at 20:49 UTC; the
owner returned to the desktop at 20:50 (seat0 `IdleHint=no`, top foreign clients gnome-shell and ghostty). All
60 timed runs were formally valid, with foreign engine time 3.6 to 9.7 % per run (0.00 % in the first campaign's
idle sessions):

| cell | parent | topic, same environment | A/A | expected (`s6-final`) | parent vs expected | arm spreads |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 11770.10 | 12118.30 | **+2.96 %** | 12962.00 | **-9.2 %** | 2.3 / 4.2 % |
| 1B 8da4w | 14027.40 | 14222.20 | +1.39 % | 15170.40 | **-7.5 %** | 7.5 / 6.0 % |
| 3B 4w | 4807.51 | 4830.19 | +0.47 % | 5184.81 | **-7.3 %** | 1.4 / 3.1 % |
| 3B 8da4w | 6023.53 | 6023.53 | 0.00 % | 6400.00 | **-5.9 %** | 2.7 / 2.7 % |
| 8B 4w | 2169.49 | 2181.04 | +0.53 % | 2306.31 | **-5.9 %** | 1.4 / 2.2 % |
| 8B 8da4w | 2817.06 | 2813.19 | -0.14 % | 2998.54 | **-6.1 %** | 1.9 / 2.4 % |

Every cell is outside the 3 % baseline tolerance and one A/A cell is outside +-2 %, so by the rules fixed in
`tools/thresholds.txt` nothing is optimised until the cause is found. The cause is the desktop in use, not the
build: the parent snapshot `s0-parent-verify`, taken 40 minutes earlier on the idle desktop with the same
binaries, read 12962 / 15170.4 / 5197.97 / 6380.06 / 2288.27 / 3002.93 tok/s (single runs), within 0.8 % of the
expected values, and the loss per run is of the size of its foreign engine share. The session is not used as
the calibration session (it would have set `BUSYMAX` to 15.62 %); it, the calibration files it wrote and the
three screen runs made in that period are kept in `.artifacts/superseded/desktop-in-use-20261008T2050Z/` (run
table copied to `results/` at the next collection). The rules are unchanged; what was added, before any
candidate was timed (`thresholds.txt`, dated block): the calibration session is `s1-aa2`; a timed run, a trace
run and a screen run start only on an idle desktop (`host.sh idle_wait`, called from `e2e5.sh`, `session.sh`,
`trace.sh`, `screen_sdpa.sh`); and the reading of the screen rule's incumbents.

The first actor run of this campaign was stopped at 19:22:50 UTC, two and a half minutes into its build of tag
`parent`; that build died with it at about 70 % and is kept, unused, under
`.artifacts/superseded/parent-build-interrupted-20261008T1922Z/`. Nothing was measured with it. Its uncommitted
draft of the kernel was read line by line against `fused3sb` and kept, with a run-time check of the
one-subgroup assumption added; the node, the profile and the tests are new.

## Done so far (no timed measurement)

- Change directory, tools copied and adapted, thresholds fixed (`7112e8930`).
- Release-zone hooks, each its own cherry-picked commit: `fab9606c3` (softmax variant name, D4.1, from
  `b969e8f1c2`) and `0ffc84a2d` (fused attention entry point, D4.3, from `1c8861aa7e`).
- Builds, each from an export of one commit, shipped SPIR-V golden PASS (53 variants) on both:
  `build/parent2` = `51d9d757f`, `build/topic1` = `9baf2de3f` (hooks + candidate 1 sources).
- **Hook condition D4 (nothing selected, nothing changed): met.** (1) `test_sarc_select` built from the parent
  export and from the branch head gives the same output for the release tables (1240 checks, 31 rows) and with
  the dev zone and `ET_VK_SARC_UNVERIFIED=1` (1562 checks, 37 rows, 213 candidates): `.artifacts/raw/d4/select-*.txt`.
  (2) `spirv_golden.py`: PASS, 53 shipped variants, on `parent2` and on `topic1`. (3) Unmodified `verify.sh`
  with no environment on `topic1` (`s0-topic1-noenv`) against the parent's (`s0-parent-noenv`): identical line
  by line with the rates removed, 34 lines, dispatched kernel names included (`tools/verify_diff.py`,
  `stage/s0-topic1-noenv/verify_diff.txt`); both `CONTROL_RECORDED` with the same five device-status items.
- Parent snapshot `s0-parent-verify` (parent environment `b580-refine3`): `CONTROL_RECORDED`, one device-status
  item (`correctness rc=1` with 28 of 28 numeric and 4 of 4 rank-3 cases PASSED); SDPA tiers 4 / 8 / 4 cases with
  0 mismatches on the parent's cooperative-matrix kernels.
- Candidate 1 sources: `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.{glsl,yaml}` (the `fused3sb` kernel for the
  8 x 16 x 16 matrix shape, packed form, one-pass and two-pass variants), `sarc_dev_b580_sdpa_kvt.{glsl,yaml}`
  (the 780M's copy pass, unchanged), `impl/sarc_dev/b580/SdpaB580Fused.cpp` (the node), a `b580-fused` block in
  `impl/sarc_dev/Overrides.cpp` (profile `b580-fused1` = `b580-refine3` + one fused variant per head_dim: a
  single name, no second variable; `b580-fused1p2` the two-pass pair; `b580-fused-<variant>` screening
  profiles), and the microbench's fused-kernel bookkeeping with the 780M's `peaked` and `fused` correctness
  tiers. `sarc/tools/check.sh --no-build`: PASS.

### Candidate 1, smoke pass (build `topic1`, `b580-fused1`, one pass per tier; `.artifacts/raw/c1-smoke/`)

`b580-fused1` = `d64_t16x32s16m8ro` (head_dim 64) and `d128_t16x64s16m8ro` (head_dim 128): 16 rows per
workgroup, one lane per row, one pass (running row maximum), Q tiles in registers. Not a gate (the gate runs 12
passes on the staged binaries).

| tier | cases | mismatches | fused kernel the only attention kernel, `pairing=ok` |
|---|---:|---:|---|
| `all` | 4 of 4 PASSED | 0 | yes |
| `extended` | 8 of 8 | 0 | yes |
| `full` (S = 2048, the three head configurations; S = 1024 at input_pos 1024) | 4 of 4 | 0 | yes |
| `peaked` (sharp rows; exercises the rescale of the one-pass form) | 5 of 5 | 0 | yes |
| `fused` (S = 64, 192, 320: shapes the 128-row QK^T tile does not take) | 4 of 4 | 0 | yes |

Error against the fp32 CPU reference in that pass, tier `full`: rms 2.05e-5 / 2.05e-5 / 2.02e-5 and maximum
7.2e-4 / 7.1e-4 / 7.9e-4 for the 1B / 3B / 8B head configurations, 8.96e-6 / 7.2e-5 at input_pos 1024. The
first campaign measured 2.76e-5 to 2.79e-5 / 1.06e-3 to 1.19e-3 and 1.29e-5 / 1.24e-4 for the parent's kernels;
the side-by-side run on the same binary (`tools/sdpa_ref.sh`) belongs to the gate and has not run yet.

## Shared-memory reading of the fused kernel (R7; the final form, written before its gate)

Read on `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.glsl` at `247d08851`, multi-subgroup form (`MULTI_SG`), one-pass
(`ONLINE`), as `b580-fused1` builds it. A workgroup is G subgroups of 16 lanes (G = 4 or 8). Lane
`L = gl_SubgroupID * 16 + gl_SubgroupInvocationID` owns row `L % 16` and segment `L / 16` of every block: a
bijection between the G x 16 invocations and the (row, segment) pairs. Every barrier is
`memoryBarrierShared(); barrier();` (`SYNC()`), a workgroup barrier.

| slot | writer | reader | what orders the read after the write |
|---|---|---|---|
| `Psh` scores, tile (i, j) | one `coopMatStore` by the subgroup that owns column j (`j % G == gl_SubgroupID`); tiles are disjoint ranges of `Psh` | each lane reads its own segment of its own row | `SYNC()` right after `qk_block` |
| `Rsh[row * SEGS + seg]` (block maximum; at the end the row's partial sum) | the one lane that owns (row, seg) | every lane of the row reads all `SEGS` slots | `SYNC()` between the store and the reads, and another after the reads before the slot is written again |
| `Gsh` ("some row's maximum rose") | any lane, only with `atomicOr(Gsh, 1)`; cleared by lane 0 alone | every lane reads it once to decide the rescale | `SYNC()` after the `atomicOr`s and before the read; lane 0 clears it only after the `SYNC()` that follows the divisor stores, i.e. after every read; the next `atomicOr` is two `SYNC()`s later |
| `Dsh[row * 4 + j]` (rescale divisors) | the lane of the row whose segment is j (segments 0 to 3): one writer per slot, and the lanes of a row hold the same value | `coopMatLoad` of the divisor tiles by every subgroup | `SYNC()` after the stores, `SYNC()` after the loads |
| `Psh` e values, own segment | the one lane that owns (row, seg) | `coopMatLoad` in `av_block` by every subgroup | `SYNC()` before `av_block`, `SYNC()` after it (before the next block's score stores) |
| `Psh[row * P_STRIDE + j]`, final divisors | the lane of the row whose segment is j | `coopMatLoad` of `den` by every subgroup | `SYNC()` after the stores; the last loads of `Psh` are a `SYNC()` earlier |
| `t_output` | one `coopMatStore` per tile by the subgroup that owns that head_dim slice | none in this kernel | |

No slot has two plain writers in a phase; the only slot several lanes write is `Gsh`, atomically and with the
same value. Control flow around every `barrier()` is uniform for the workgroup: `num_blocks` and `s_base` are
per workgroup, the two early returns come before the first barrier, and the rescale branch is taken on the
value of `Gsh` that every lane reads after the same barrier (that is why the vote of the single-subgroup form,
`subgroupAny`, is replaced). The accumulators and the Q tiles are per-subgroup registers and are never shared.
The assumption "the workgroup is exactly G full subgroups of 16": the pipeline is created with required
subgroup size 16 (yaml `SUBGROUP_SIZE`, as for the Xe2 kernels) and the node launches a local size of G x 16;
the release-zone pipeline code does not set the full-subgroups flag, so the kernel checks
`gl_NumSubgroups == G` and `gl_SubgroupSize == 16` and writes NaN rows otherwise (one writer per row). The tiers' mismatch count
would not report them (corrected: "Finding F2"); the `[sdpa-error]` record would, and the gate now requires it
finite. The copy pass `sarc_dev_b580_sdpa_kvt` has no shared memory.

## Specification quotes for the shared-memory reading (owner note 2026-10-09 00:20 UTC; looked up 01:25 UTC)

From the `vulkan-docs` MCP server (pages under `https://docs.vulkan.org`). The sentences were collected by a
subagent of the actor session reading the index; four of them (marked *) were searched again as exact phrases by
the actor and found on the page named. **No sentence found contradicts the reading above; two points are not
covered by a normative sentence in the index, and one requirement of the specification is not met by the
pipeline code of this branch, the parent's kernels included (finding F1 below).**

1. `barrier()` and writes to `shared` variables.
   - * "For any given static instance of barrier(), all tessellation control shader invocations for a single
     input patch must enter it before any will be allowed to continue beyond it, or all compute shader
     invocations for a single workgroup must enter it before any will continue beyond it."
     (`glsl/latest/builtinfunctions.md`, /glsl/latest/chapters/builtinfunctions.html)
   - "A barrier() affects control flow but only synchronizes memory accesses to shared variables and
     tessellation control output variables." (same page)
   - "In compute shaders, barrier() is equivalent to controlBarrier() with execution and memory scope equal to
     gl_ScopeWorkgroup, storage semantics equal to gl_StorageSemanticsShared, and sem equal to
     gl_SemanticsAcquireRelease." (`glslext/latest/GL_KHR_memory_scope_semantics.md`)
   - Not normative (tutorial `04_memory_consistency.md`): a `memoryBarrierShared()` before `barrier()` "is
     redundant". So `SYNC()` = `memoryBarrierShared(); barrier();` orders as the reading assumes through the
     `barrier()` alone; the first call adds nothing and is kept (it is the 780M kernel's form).
2. Uniform control flow.
   - "For compute shaders, the barrier() function may be placed within control flow, but that control flow must
     be uniform control flow." and "Otherwise, some shader invocations will stall indefinitely, waiting for a
     barrier that is never reached by other invocations." (`glsl/latest/builtinfunctions.md`)
   - Checked again in the kernel at `247d08851`: both early returns come before the first `SYNC()` and depend
     only on `gl_WorkGroupID`, uniforms, `gl_NumSubgroups` ("uniform across the invocation group",
     `GL_KHR_shader_subgroup`) and `gl_SubgroupSize`; the rescale branch depends on `Gsh` read after a `SYNC()`.
3. Atomics and data races.
   - "Atomic memory functions perform atomic operations on an individual signed or unsigned integer stored in
     buffer object or shared variable storage." and "The contents of the memory being updated by the atomic
     operation are guaranteed not to be modified by any other assignment or atomic memory function in any shader
     invocation between the time the original value is read and the time the new value is written."
     (`glsl/latest/builtinfunctions.md`)
   - "Let X and Y be operations that access overlapping sets of memory locations M, where X != Y, and at least
     one of X and Y is a write, and X and Y are not mutually-ordered atomic operations. If there does not exist
     a location-ordered relation between X and Y for each location in M, then there is a data race.
     Applications must ensure that no data races occur during the execution of their application."
     (`spec/latest/memorymodel.md`, /spec/latest/appendices/memorymodel.html)
   - Consequence for `Gsh`: the concurrent `atomicOr`s are atomic operations; the plain clear by lane 0 and the
     plain reads are each separated from them by a `SYNC()` (table above), as the second quote requires.
4. Subgroups of the workgroup.
   - "If the shader was created with a required subgroup size, the SubgroupSize decorated variable will match
     that value." (`refpages/latest/SubgroupSize.md`)
   - * "Full subgroups are required when the X dimension of the workgroup size is a multiple of the reported
     value of the SubgroupSize built-in, and one of the following is true: The shader was created from SPIR-V
     with version number of 1.6 or higher. The shader was created with vkCreateShadersEXT and the
     VK_SHADER_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT flag was set. The shader was created as part of a pipeline
     and the VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag was set." (`spec/latest/shaders.md`)
   - "If full subgroups are not enabled, some subgroups may be dispatched with inactive invocations that do not
     correspond to a local workgroup invocation, making the value of index unreliable." and "There is no direct
     relationship between SubgroupLocalInvocationId and LocalInvocationId or LocalInvocationIndex."
     (`refpages/latest/SubgroupLocalInvocationId.md`)
   - What this means here: the fused shaders are SPIR-V 1.3 (header word `00010300` in `build/topic6`; the
     instance asks for Vulkan 1.1, `vk_api/Runtime.cpp`) and the pipeline does not set the flag, so **the
     specification does not guarantee full subgroups for this kernel**; the run-time check is what the
     kernel's arithmetic rests on (it does not make the pipeline valid: finding F1). For the arithmetic it is
     sufficient: the local size is G x 16 in X (`SdpaB580Fused.cpp`, `LocalWorkGroup(wg, 1, 1)`),
     and `gl_NumSubgroups == G` with `gl_SubgroupSize == 16` leaves no room for a subgroup with fewer than 16 of
     the workgroup's invocations. The kernel indexes lanes by `gl_SubgroupID` and `gl_SubgroupInvocationID`
     only, never by `gl_LocalInvocationIndex` (except in the NaN fallback, where any `WG_TILE_M` distinct
     invocations do).
5. Cooperative-matrix loads and stores on `shared` arrays.
   - "VUID-StandaloneSpirv-Pointer-08973 The Storage Class of the Pointer operand to OpCooperativeMatrixLoadKHR
     or OpCooperativeMatrixStoreKHR must be limited to Workgroup, StorageBuffer, or PhysicalStorageBuffer"
     (`refpages/latest/StandaloneSpirv.md`)
   - * "For each memory location accessed by a dynamic instance of a cooperative matrix store instruction
     (...), a single implementation-dependent invocation within the instance of the matrix's scope performs a
     non-atomic store to that memory location." and, for loads, "some implementation-dependent invocation(s)
     within the instance of the matrix's scope perform a non-atomic load from each memory location that is
     defined to be accessed by the instruction." (`spec/latest/memorymodel.md`)
   - "VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986 For OpCooperativeMatrixLoadKHR and
     OpCooperativeMatrixStoreKHR instructions, the Pointer and Stride operands must be aligned to at least the
     lesser of 16 bytes or the natural alignment of a row or column (...)" (`refpages/latest/RuntimeSpirv.md`).
     Met: `Psh` is `uvec4[]` and `Dsh` `vec4[]`, so every element offset and stride is a multiple of 16 bytes.
   - So a tile store is one plain store per location and a tile load plain loads: disjoint tiles stored by
     different subgroups do not overlap, the same tile loaded by several subgroups has no writer, and every
     load of a tile another subgroup stored follows a `SYNC()` (table above). **Not covered by a sentence in the
     index:** that all invocations of the subgroup must execute a load or store with the same operands (the
     SPIR-V extension specification is not indexed and the built-in function section of the GLSL extension page
     is cut short). The kernel's operands depend only on loop counters and `gl_SubgroupID`, equal within a
     subgroup.

**Finding F1 (a validity defect of the two new fused pipelines as well).**
* "VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770 Any pipeline containing a shader with
OpTypeCooperativeMatrixKHR or OpCooperativeMatrix*KHR instructions must be created with the
VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be version 1.6 or
greater" (`refpages/latest/RuntimeSpirv.md`). Every cooperative-matrix pipeline of this branch is SPIR-V 1.3
without the flag: the shipped release kernels, the parent's Xe2 attention kernels and the fused kernel alike
(checked on `sarc_sdpa_qk_coopmat_xe2c_...` and both fused variants of `build/topic6`). Setting the flag is a
change of the release-zone pipeline code (`vk_api/`), which decision D4 does not cover, and SPIR-V 1.6 needs an
instance of Vulkan 1.3; neither was done. ANV launches full subgroups for these kernels in every run measured
(the fused kernel would have written NaN rows otherwise, and every `[sdpa-error]` record of
every gate is finite: "Finding F2"). **Decided by the owner at 15:25 UTC: "Known defect: F1" below.**

## Known defect: F1 (owner decision 2026-10-09 15:25 UTC, "F1: C")

The owner's ruling: no release-zone change for F1 in this campaign; neither form below is authorised here. F1
is a known defect shared with every cooperative-matrix pipeline of every device, the shipped ones included
(this campaign's parent has it; so do the fused ports closed on 2026-10-09 on the RTX 4070 Ti SUPER, the Radeon
780M and the Arc Pro B70). It will be repaired once, in the release zone, in a change of its own before any
promotion pull request, with every device gated and timed again under it. **Every number of this campaign
stands as measured and is a number of pipelines created without the full-subgroups flag (F1).**

- The specification sentence: "VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770 Any pipeline containing a
  shader with OpTypeCooperativeMatrixKHR or OpCooperativeMatrix*KHR instructions must be created with the
  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be version 1.6 or
  greater" (`refpages/latest/RuntimeSpirv.md`, `vulkan-docs` MCP server).
- The SPIR-V version: 1.3 (header word `0x00010300`) for both selected fused shaders,
  `sarc_dev_b580_sdpa_fused_d64_t16x64s16m8g4roj` and `..._d128_t16x128s16m8g8oj`, in `build/topic7` (and
  `topic6`), and for the parent's `sarc_sdpa_qk_coopmat_xe2c_...`.
- The code: `backends/vulkan/runtime/vk_api/Pipeline.cpp` creates every compute stage with `flags` `0u`, at line
  305 (`ComputePipeline`) and line 540 (the batched pipeline creation); the required subgroup size is chained
  in `pNext` just above each. `backends/vulkan/runtime/vk_api/Runtime.cpp:91` creates the instance with
  `VK_API_VERSION_1_1`, so SPIR-V 1.6 (Vulkan 1.3) is not available either. The runtime already reads
  `computeFullSubgroups` (`vk_api/Device.cpp:317`); the B580 reports it.
- What the kernel's run-time check guarantees: it compares `gl_NumSubgroups` with G and `gl_SubgroupSize` with
  16 before the first barrier; with a local size of G x 16 in X that leaves no room for a subgroup holding
  fewer than 16 of the workgroup's invocations, so whenever the kernel computes, its lane-to-(row, segment)
  mapping is the bijection the shared-memory reading assumes. If the check fails, the rows are written as NaN
  instead of being computed.
- What it does not guarantee: it does not make the pipeline valid Vulkan (the requirement is on pipeline
  creation; a driver may reject or mis-handle such a pipeline without the kernel ever running); it relies on the
  two built-ins being meaningful in a pipeline the specification does not cover; it says nothing about the
  parent's and the shipped cooperative-matrix kernels, which have no such check; and its NaN rows are not
  counted by the correctness test's mismatch counter ("Finding F2"). ANV launched full subgroups in every gated
  run: all 960 `[sdpa-error]` records are finite.
- The two forms put to the owner, as input to the later change:
  - **Form A, per shader, inert unless a shader asks for it.** A yaml parameter (say `FULL_SUBGROUPS: 1`) that
    `gen_vulkan_spv.py` passes into `ShaderInfo` beside the required subgroup size, a field in the pipeline
    descriptor and its hash / equality, and in `Pipeline.cpp` the stage flag
    `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` when the field is set, a required subgroup
    size is in force and the device reports `computeFullSubgroups`. Files:
    `backends/vulkan/runtime/gen_vulkan_spv.py`, `vk_api/Shader.{h,cpp}`, `vk_api/Pipeline.{h,cpp}`, and where
    the descriptor is filled; an estimated 30 to 40 lines. Shaders that do not set it keep their pipelines.
  - **Form B, for every pipeline with a required subgroup size.** In `Pipeline.cpp` at both places, set the
    flag whenever a required subgroup size is in force and the device reports `computeFullSubgroups`; about 10
    lines in one file plus the feature bit in the descriptor. It repairs the shipped and the parents'
    cooperative-matrix pipelines too, and changes how each of them is created: with the flag, the local size
    in X must be a multiple of the required subgroup size (VUID-VkPipelineShaderStageCreateInfo-pNext-02757).
- Decode with the fused node present (not investigated, as ordered). Decode does not run the fused kernel.
  Final / parent, 32 tokens, medians of 5, tok/s, `s4-final` (`topic7`, idle desktop):

  | cell | parent | final | ratio | `s2-c1` ratio (`topic6`, desktop in use) |
  |---|---:|---:|---:|---:|
  | 1B 4w | 96.8750 | 95.3846 | 0.98462 | 0.99396 |
  | 1B 8da4w | 88.8252 | 88.0682 | 0.99148 | 0.99448 |
  | 3B 4w | 44.2857 | 43.9093 | 0.99150 | 0.99177 |
  | 3B 8da4w | 40.7359 | 40.3121 | 0.98960 | 0.99621 |
  | 8B 4w | 26.4731 | 26.4280 | 0.99830 | 0.99918 |
  | 8B 8da4w | 24.7406 | 24.6423 | 0.99603 | 0.99533 |

  Recomputed from `stage/{s4-final,s2-c1}/decode/decode.csv` (medians of the five per-run rates, ratios of the
  unrounded medians). An earlier version of this table gave three decimals taken from the rounded summary and
  had 0.991 for `s4-final` 3B 4w (0.99150, i.e. 0.992) and 0.995 for `s2-c1` 1B 8da4w (0.99448, i.e. 0.994).

  0.2 to 1.5 % slower in `s4-final` and 0.1 to 0.8 % in `s2-c1`: inside the +-2 % band in every cell, and below 1
  in all twelve readings (the B70 confirmation measured 0 to 2.5 %).

## Next

Nothing in this campaign. For the later release-zone change that repairs F1: one of the two forms above, the
microbench's NaN-blind mismatch counter ("Finding F2"), then every device gated and timed again.

## Decision needed from the owner

None. Finding F1 was decided on 2026-10-09 15:25 UTC (option C, section "Known defect: F1"); items 2 and 3 of the
earlier list on 2026-10-09 09:55 UTC (`thresholds-history.md`).

## Blocking

Nothing.
