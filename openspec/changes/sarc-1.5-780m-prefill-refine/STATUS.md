# STATUS: 780M prefill campaign, round 2 (parameter space + beyond)

Updated 2026-10-05 18:15 PDT (2026-10-06 01:15 UTC). Parent for this round: profile `780m-refine3` (build `topic-r1`).
Artifacts: `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/` (new raw data) and
`.../780m-prefill-refine-2026-10-03/` (earlier builds and sessions).

## State

| item | state |
|---|---|
| candidate 7 (softmax r3) | **complete**: gate passed, bit-identical to its parent, six cells +2.39 to +6.78 %, geomean **+4.10 %** over `780m-refine3` |
| candidate 8 (fused SDPA kernel, Part 2) | **ACCEPTED (reference-error rule, owner decision 2026-10-04)**: +7.84 to +20.14 %, geomean **+13.16 %** over candidate 7, 60 valid runs; gate finished 00:04 PDT; one next-token item differs (8B 8da4w, `prompt_2048.txt`); record below. It needs `hooks/sdpa-fused-hook.patch`, so adopting it is the owner's decision |
| igpu-roofline `fast` plan | finished 20:15 PDT; matrix roofs 14.766 TFLOP/s (fp16 -> fp32) and 14.379 TOP/s (int8) |
| candidate 9 (candidate 8 + one-pass fused kernel + 4w kernel per shape) | **gate passed** (finished 04:10 PDT): +4.50 to +5.45 % in the 4w cells, +1.06 to +1.80 % in the 8da4w cells (inside the band), geomean **+3.19 %** over candidate 8; every next-token item SAME; record below |
| 4w, Part 1 | random sample, refinement round 1 and the confirmation are done: **best kernel per shape below**, 4.0 to 4.8 % less linear time per layer than `780m-refine3`, byte-identical output; refinement round 2 (51 neighbours of the winners) and the 12 production-diff passes are queued |
| candidate 10 (candidate 9 with the refined 4w table, profile `refine10`) | **gate passed, +0.47 % geomean: inside the band, not a gain** (4w cells +0.94 / +1.43 / +1.16 %, 8da4w cells 0.00 / -0.73 / +0.06 %); byte-identical 4w output; first candidate under 2 % |
| 4w, Part 1 | finished except for the production-diff passes: the search stopped moving after refinement round 3 and the geometry scan; the best kernel per shape (5 repeats) is profile `refine10` |
| 8da4w | screening mode validated on 64 configurations (rank correlation 0.998 to 1.000, the full top 10 inside the screen's top 20 on all three models); screen of the 2,238 survivors: 261 done when the host hung at 08:53 PDT, continued after the reboot |
| production-diff passes; QK^T / attn*V enumeration | queued behind the 8da4w steps |
| release-zone hooks (owner decision 2026-10-05) | committed (`b969e8f1c`, `1c8861aa7`, dev side `8518659ef`); candidate 10 reproduced from the committed build: gate passed, **+22.64 %** geomean over `780m-refine3` measured directly (section "Committed build") |

Chained over the three sessions (`t1-recheck`, `c7-softmax-r3`, `c8-fused3`; not a direct measurement, that comes
at the end): candidate 8 is about +27 % geomean over `dev/1.5` (1B 4w 2698 -> 3625 tok/s, 8B 8da4w 488 -> 606).

## Coordinator hold (owner decision 2026-10-06 00:35 UTC): in place since 2026-10-05 17:50 PDT

- The coordinator creates and removes **`/home/doremy/hmz-sarc/.artifacts/HOLD`** (on `rocky-ryzen`).
- The queue answers with **`/home/doremy/hmz-sarc/.artifacts/HELD`**: one line
  `HELD <UTC time> <what it would start next>` per waiting queue, written only when nothing of the campaign is
  running; removed when `HOLD` is gone. While `HOLD` exists nothing new starts; the poll is once a minute; a unit
  that is running when `HOLD` appears finishes normally; afterwards the queue continues where it was (every step
  is resumable from its CSV).
- Mechanism (`tools/hold.sh`): every unit holds a shared lock on `.artifacts/.hold-busy` while it runs and looks
  at `HOLD` again after taking it (and after taking the gpu-lab lock), so `HELD` can only be written when an
  exclusive lock on that file is free: no unit is running and none can start. A watcher (`hold.sh watch`,
  detached, pid in `<artifacts 10-05>/logs/hold-watch.out`'s process) writes `HELD` when `HOLD` appears while the
  queue is idle or waiting on a marker, so the coordinator is never left waiting.
- Units, and how long one lasts at most:

  | unit | where the hold is checked | lasts |
  |---|---|---|
  | one sweep configuration (`sweep_space.py`: screen, confirmation timing, SDPA enumeration) | before every configuration | 3 to 40 s; hard limit 600 s per process |
  | one microbench job through `gl.sh` (production-diff pass, SDPA correctness pass, SDPA perf run) | before every job | seconds; up to 3.5 min (`--sdpa-tier=full`) |
  | one warm trace set (`trace.sh`, 12 runs) | at its start | about 5 min |
  | one `verify.sh` (not editable, so wrapped as a whole) | at its start | 8 to 15 min |
  | one build (`build-both.sh`, `build_space.sh`; CPU only) | at its start | 2 to 25 min |
  | **one timed session** (`e2e5.sh`: 5 valid runs x 6 cells x 2 arms, interleaved) | at its start; the cool start is taken again after a hold | 35 min measured, **at most about 50 min: the longest unit** |

  A timed session is not split: a foreign workload between its interleaved runs would make it a different
  session, and it could not be resumed without discarding it.
- The existing `PAUSE` marker is **not** this mechanism and is unchanged: it lives in the dated directory, is set
  by the campaign's own gates against its own sweep, writes no `HELD`, and the confirmation steps ignore it.
  `HOLD` is checked in addition, by everything, and `--ignore-pause` does not skip it.
- Tested 2026-10-06 00:44 to 00:46 UTC with the test name (`SARC_HOLD_NAME=HOLD-TEST`, same code path, 2 s
  poll): a unit running when `HOLD-TEST` appeared finished; no `HELD-TEST` while it ran; then
  `HELD 2026-10-06T00:44:40Z test unit B (would start next)`; unit B did not start until `HOLD-TEST` was removed,
  then ran within 1 s and `HELD-TEST` was gone. Watcher alone: `HELD-TEST` with "queue idle" after 5 s, removed
  when the test file was removed. The sweep's Python path was tested the same way (below, "Running now").
- The running queue picked it up at a job boundary without being killed: the confirmation repeat that was running
  (`full-r4`, started before the change) finished with the old code; `full-r5` and everything after it start as
  new processes with the hold. While `HELD` exists I start no GPU job and no build by hand either.

## Host hang and reboot, 2026-10-05 08:53 PDT (recorded 10:45 PDT, before anything was restarted)

The host stopped responding at 08:53 PDT (last journal entry of that boot 08:53:42) and was rebooted by hand at
09:34 PDT. Same kernel (6.12.0-211.61.1.el10_2), same driver (RADV PHOENIX, Mesa 25.2.7), GPU at `auto` with the
800 / 1100 / 2799 MHz levels as before.

**Owner decision 2026-10-05: no profiler captures on this device.** The RGP request of the same morning is
withdrawn. No `MESA_VK_TRACE*`, `RADV_THREAD_TRACE_*` or `RADV_PROFILE_PSTATE` in any process; `tools/gl.sh`
refuses to start a job that carries one. Evidence stays ETDump, shader-clock phase timing, kernel timing and
igpu-roofline.

What was running when the host went down (from `<artifacts>/rgp/`, `raw/dq/screen.{csv,out}`,
`logs/sweep-pauses.txt`; `<artifacts>` = `.../780m-prefill-refine-2026-10-04`):

| job | state at 08:53 | lost |
|---|---|---|
| `rgp/take.sh` (RGP captures with `MESA_VK_TRACE=rgp`, per submit, 512 MB thread-trace buffer, cache counters; one microbench SDPA run each, under the gpu-lab lock) | captures 1 and 2 (three kernels, release softmax and softmax r3) finished at 08:53:17 and 08:53:23. Capture 3 (fused two-pass) started 08:53:26 and its run returned rc = 0; capture 4 (fused one-pass) had started (its three files are stamped 08:53:31 to 08:53:33) when the machine hung | the files of captures 3 and 4 are zero-length (never reached the disk). Nothing from any capture is used anywhere in this change |
| `chain6e.sh`: 8da4w screen of the 2,238 survivors, sharing the GPU run by run with the captures | 261 configurations done, last row 08:53:25 | the process. `raw/dq/screen.csv` is intact (783 rows, 21 fields each, all dispatched); the sweep resumes from it |
| queued behind it in the same chain: 8da4w confirmation, production-diff passes, QK^T / attn*V enumeration | not started | the queue (restarted below) |

Not lost: every gated session (the candidate-10 gate finished at 08:49 and was committed at 08:50); the staged
binaries of `c7` to `c10` still match the sha256 in their `STAGE.md`; `git fsck` is clean; no result file written
near the hang is truncated.

Where the capture material is (kept as evidence, nothing calls it):

- `<artifacts>/rgp/take.sh.DISABLED-by-owner-2026-10-05` and `cap.sh.DISABLED-by-owner-2026-10-05`: the capture
  scripts, renamed and not executable, with `README-DISABLED.txt`. No chain script, tool or timer references
  them (checked with grep over `<artifacts>/*.sh`, both `tools/` directories; no crontab, no user timers).
- `<artifacts>/rgp/<capture>/`: the capture files and what was extracted from the first two.
  `/tmp/rgpmb_2026.10.05_08.53.3{1,2,3}.rgp`: the files RADV was writing for capture 4, left where they are.
- `<artifacts>/fx/rgpmb`: a copy of the microbench under another name (so that the capture files were named
  after it); an ordinary binary, not used by anything else.
- `tools/rgp_chunks.py` in this directory (untracked, not committed): a reader for `.rgp` files. It sets no
  variable and starts no GPU job; nothing calls it.

## Running now

**Held by the coordinator since 2026-10-06 00:53:06 UTC** (`HOLD` created; `HELD` written 00:53:12 UTC, six
seconds later, when the configuration that was running had finished). Nothing of the campaign runs on the GPU and
no build is started until `HOLD` is removed. `HELD` says what continues then:
`sweep_space.py full-r5.csv: 8da4w bt_t128x128k64g42s32afmb2` (the 8th of 22 configurations of the last
confirmation repeat).

Queue behind the hold (`chain17.sh`, `chain18.sh`, detached, all steps resumable):

1. 8da4w confirmation, repeat 5 of 5 (15 configurations left, 16 s each), then its summary;
2. 12 production-diff passes of the confirmed 8da4w and 4w configurations;
3. QK^T / attn*V enumeration at a steady clock (20 warm-up + 8 timed runs).

Prepared and not yet built (a build is a unit, so it waits for the hold too): candidate 11 = candidate 10 + the
8da4w kernel per shape (`ET_VK_SARC_780M_PROFILE=c11`), then its gate. The 8da4w screen of all 2,238 survivors
finished 17:10 PDT; with four complete repeats the best kernel per shape is 1.9 to 3.0 % less 8da4w linear time
per layer than the `780m-refine3` kernel, which predicts about +1.6 to +2.3 % in the three 8da4w cells and about
+1 % geomean: expected to be the second candidate under 2 %.

### After the reboot: re-check and A/A (`<artifacts 10-05>/stage/t4-recheck`, `t5-aa`; 10:28 to 11:38 PDT)

`t4-recheck`, pristine `dev/1.5` build against build `topic-r1` with `780m-refine3`, 5 valid runs per arm, cool
start: 1B 2694.74 -> 2832.64 (+5.12 %), 2537.79 -> 2828.73 (+11.46 %); 3B 1146.70 -> 1192.08 (+3.96 %), 1060.04 ->
1168.28 (+10.21 %); 8B 522.32 -> 545.99 (+4.53 %), 488.43 -> 549.80 (+12.56 %) (4w, 8da4w); geomean **+7.92 %**
(before the reboot: +7.91 %). Clock 2771 to 2800 MHz, all next-token items SAME. The device is as it was.

`t5-aa`, candidate 10 (build `fused9`) on both arms, 3 repeats: cells -0.09 to +0.19 %, geomean 0.00 %.
(`summary.csv` of that session says INCOMPLETE because the tool expects 5 runs; the medians are from `runs.csv`.)

### The hook commits, and the edit that was interrupted (found and fixed 11:15 to 11:43 PDT)

The control session was stopped by the owner's coordinator in the middle of the dev-zone edit that follows the
two hook commits (`b969e8f1c` softmax variant name, `1c8861aa7` fused attention node). The working tree was
inspected before anything else: the edit was textually complete but **not working**. The first build of it
(`build/wt1`) selected the softmax variant and never fired the fused node (`raw/wt1-smoke/`: `fused=-` with
profiles `c8` and `c10`, and with the explicit variable). Cause: `Sdpa780mFused.cpp`, now its own file under
`780m/`, registered its two functions from its own static initializer; the shared `Registrar` of `Overrides.cpp`
starts from an empty `Override` and ran afterwards. Fixed inside the 780m block (the pair is kept there and
applied by whichever initializer runs last), rebuilt (`build/wt2`), and checked with every profile
(`raw/wt2-smoke/`): no profile -> release kernels and release softmax; `c7` -> softmax `..._780m_r3`; `c8` ->
`fused3_..rk`; `c9`, `c10` -> `fused3_..rko`; the names of the measured builds. `check.sh --no-build` passes.
Committed as `8518659ef`. Nothing was measured with the broken build.

## Committed build: candidate 10 reproduced from the branch alone (owner decision 2026-10-05, release-zone hooks)

Build `head1` = branch head `8518659ef`, clean tree, no local patch (`<artifacts 10-05>/build/head1`). Evidence in
`results/780m/sessions/{h0-control,h1-c10}/` and `results/780m/hooks/`.

**Nothing selected: every dispatch as before.**

- `test_sarc_select`: PASS, 1240 checks / 31 rows (release tables) and 1432 checks / 121 candidates (dev zone),
  the counts of the parent. `spirv_golden.py` on `head1`: PASS (53 shipped variants).
- All 1,325 shaders of the pristine `dev/1.5` build (`build/parent`) are byte-identical in `head1`.
- `h0-control`: `head1` with no environment, unmodified `verify.sh`, against the pristine `dev/1.5` control
  (`s0-parent-verify`): the 34 lines are identical once the measured tok/s figures are set aside (kernel names,
  return codes, 12 production-diff cases ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens).
  `--sdpa-correctness-only` three times: 4 of 4, release QK^T / softmax / attn*V names, `fused=-`.

**Candidate 10 dispatches what was measured.** Selected by `ET_VK_SARC_780M_PROFILE=c10` on top of
`ET_VK_SARC_DEV_PROFILE=780m-refine3`. All 1,468 SPIR-V binaries of the measured build (`fused9`) are
byte-identical in `head1` (the six softmax variants under their new names). Kernel names in `verify.sh`
(`linear 4w` / `linear 8da4w` lines) and in the SDPA tiers (`fused3_d64_t32x32g11s32rko`,
`fused3_d128_t16x64g11s32rko`) are those of the candidate-10 gate; `verify.out` of `h1-c10` equals that gate's
line for line apart from the tok/s figures.

**Gate, once, on the committed build (`h1-c10`, 12:03 to 13:22 PDT):** SDPA tiers `all`, `extended`, `full`, 12
passes each: 48 / 96 / 48 cases passed, 0 failed, 0 mismatches, pairing ok in every line; `verify.sh` rc = 0 as
above. Next token against the parent: SAME in eleven of twelve items; 8B 8da4w on `prompt_2048.txt` DIFFERS,
the item already recorded for candidate 8 (**ACCEPTED (reference-error rule, owner decision 2026-10-04)**,
evidence `results/780m/probe/`, `results/780m/sdpa-error/`; the kernels are byte-identical, so that evidence
stands).

**Timed session against the parent (`780m-refine3`), same binary in both arms, started at 44 C, 5 valid runs per
arm.** This is also the first direct measurement of candidate 10 against the round's parent; until now the
figure was chained over four sessions (1.0410 x 1.1316 x 1.0319 x 1.0047 = +22.13 %).

| cell | `780m-refine3` | candidate 10, committed build | gain | candidate 10 through the patches (`c10-q4-refine10`) | spread parent / candidate |
|---|---:|---:|---:|---:|---|
| 1B 4w | 2828.73 | 3835.21 | **+35.58 %** | 3842.40 | 0.28 / 0.19 % |
| 1B 8da4w | 2824.83 | 3690.09 | **+30.63 %** | 3690.09 | 0.28 / 0.18 % |
| 3B 4w | 1190.01 | 1459.73 | **+22.67 %** | 1459.73 | 0.06 / 0.07 % |
| 3B 8da4w | 1166.29 | 1379.12 | **+18.25 %** | 1360.80 | 0.57 / 1.46 % |
| 8B 4w | 542.09 | 641.40 | **+18.32 %** | 642.21 | 0.16 / 0.28 % |
| 8B 8da4w | 549.50 | 615.20 | **+11.96 %** | 615.20 | 0.43 / 0.42 % |

Geomean **+22.64 %** over `780m-refine3`. Committed build against patch build: -0.19 to +1.35 %, geomean +0.17 %
(different sessions; the A/A floor is +-0.2 %, and 3B 8da4w is the cell with the 1.4 % repeat spread in both).
Traces (warm, one run per arm): 718.2 -> 536.2 ms (1B 4w), 719.2 -> 552.0, 1719.6 -> 1417.5, 1742.7 -> 1484.7,
3809.2 -> 3250.7, 3741.0 -> 3364.6 ms (8B 8da4w); in 1B 4w the three attention kernels (46.5 + 115.3 + 42.1 ms)
are gone, the copy family grows by 35.6 ms (copy pass + fused kernel are counted there), linear GEMM 366.6 ->
352.9 ms.

`sarc/tools/check.sh --no-build`: PASS. Its zone rule compares against `release/1.5`, where the release zone is
allowed and `SDPA.cpp` is already listed in `sarc/HOOKS`, so it does not flag the two entry points; they are the
two commits named above, permitted by the owner decision of 2026-10-05 and by nothing else.

### What the clock does to the microbench (found 2026-10-04, affects how the sweeps are read)

The GPU clock idles at 800 MHz and needs sustained load to reach 2800 MHz. In `llama_main` it is at 2800 MHz
throughout; in the microbench it is not always:

- SDPA suite, 3 warm-up + 5 timed runs of 3 to 15 ms each: the timed runs are still on the rising clock. The
  coefficient of variation over the 5 runs is 7 to 9 % (median of the 330 enumeration rows measured so far) and a
  compute-bound kernel reads up to 1.7 times slower than at 40 + 10 runs, while a memory-bound one barely moves.
  `ET_VK_SDPA_PERF_RUNS` (added) sets the run counts; the enumeration will use 20 + 8. The 55 runs done with 3 + 5
  stay in `raw/space/results.csv` and are not used for ranking.
- Linear suite, full measurement (3 + 5): steady, coefficient of variation 0.5 to 1 %. Longer runs only heat the
  device (300 + 5: +5 %).
- Linear suite, screening mode (1 + 2): the first timed run is still rising (12 to 15 % between the two timed
  runs). On 14 configurations across the ranking the screening value is 3 to 13 % above the steady one, by a
  different amount per configuration; the order is kept overall (Spearman 0.987) but not among the first five
  (`results/780m/space/clock-check.csv`). The random sample and refinement round 1 were screened this way, so their
  single values scatter by about +-5 %: fine for the parameter-importance table (averages over many
  configurations), not for choosing among near-ties. Therefore the confirmation takes everything within 8 % of the
  fastest (not the 10 best by rank), and the 8da4w screen uses 3 + 3 runs.

### Pauses and overlaps (for whoever audits the timings)

- Builds are CPU work and the screening sweep does not look at `PAUSE`, so the sweep process was stopped
  (`SIGSTOP`, between two runs) for each build and continued afterwards: 03:33:14 to 03:34:33, 03:41:31 to
  03:41:33, 03:41:53 to 03:43:11, 03:49:09 to 04:05:35 and 04:13:42 to 04:13:45 UTC (`logs/sweep-pauses.txt`).
- Candidate-8 correctness and kernel-timing runs took the gpu-lab lock between sweep runs from 03:28 to 04:16 UTC.
  They cannot overlap a sweep run, but they warm the device, and the sweep waits until it is back under 62 C
  before each run: that is why round 1 is slow, not a change in what it measures.
- During the candidate-8 gate: a build (05:29 to 05:34 UTC) and correctness / kernel-timing runs of the one-pass
  variant and the four-arm logits run (05:35 to 05:57 UTC) were slotted in while the gate ran its SDPA
  correctness tiers, after the timed session had ended (05:28 UTC). Nothing ran next to the session.
- The gate of candidate 9 was first started at 08:59 UTC and did not run: the sweep process had been stopped
  while it held the gpu-lab lock, so the session timed out on the lock after 15 min (`gpu-lab lock busy`).
  Nothing was measured; the empty outputs are in `<artifacts>/superseded/c9-gate-lock-held-by-stopped-sweep/`.
  It was restarted at 09:18 UTC with the sweep stopped while the gate script itself held the lock.
- One build of mine started while the roofline was still running and was stopped after about 15 s (03:05:28 to
  03:05:42 UTC, `-j6`). It overlapped the confirm runs of `mem_write`, `mem_copy`, `mem_triad` and
  `sharedbw_fp16`. The matrix roofs used below were confirmed earlier in the run and repeat within 0.1 %; the
  shared-memory fp16 read / write roofs show a 15 % repeat range and are not used.

## Roofs, re-measured (igpu-roofline `fast` plan, 2026-10-04 19:46 to 20:15 PDT)

Run `~/igpu-roofline/campaigns/780m/2026-10-04-fast-prefill-refine2` (code `dbdd193e`, RADV PHOENIX, Mesa 25.2.7,
GPU clock DVFS-governed, not pinned). Confirmed medians (3 repeats each):

| roof | value | repeat range | earlier value (`sarc-1.5-e2e-benchmark/evidence/roofline.md`) |
|---|---:|---:|---:|
| `matrix_fp16_fp32` (4w linear, SDPA) | 14.766 TFLOP/s | 0.1 % | 14.772 |
| `matrix_int8` (8da4w linear) | 14.379 TOP/s | 0.1 % | 14.393 |
| `global_read` / `global_write` / `global_copy` | 86.6 / 77.6 / 71.5 GB/s | 0.0 to 0.1 % | |
| MMA fed from shared memory, one tile pair per 1 / 2 / 4 / 8 multiply-adds | 2.42 / 4.82 / 9.65 / 14.73 TFLOP/s | | |
| MMA fed from a cache-resident buffer, same | 4.01 / 7.89 / 10.75 / 12.80 TFLOP/s | | |

The last two rows are what shaped candidate 8: operand loads, not the matrix unit, limit a kernel that loads a
tile pair for fewer than about 8 multiply-adds, and contiguous tiles in a cache-resident buffer load faster than
tiles in shared memory.

## Candidate 7 (softmax r3, through the uncommitted hook): ACCEPTED, bit-identical, +4.10 % geomean

Session `c7-softmax-r3` (`results/780m/sessions/c7-softmax-r3/`, raw data `<artifacts>/stage/c7-softmax-r3`),
started 18:13 PDT at 45 C after the full 30 min wait. Both arms are the same binary (build `hook3` = the branch +
`hooks/softmax-name-hook.patch` in a scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3`; the candidate arm
adds `ET_VK_SARC_780M_SOFTMAX=r3`. Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (`780m-refine3`) | candidate (+ softmax r3) | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 2828.73 | 3020.65 | **+6.78 %** | 0.28 / 0.29 % | SAME / SAME |
| 1B 8da4w | 2824.83 | 3011.76 | **+6.62 %** | 0.28 / 0.15 % | SAME / SAME |
| 3B 4w | 1190.70 | 1231.51 | **+3.43 %** | 0.12 / 0.18 % | SAME / SAME |
| 3B 8da4w | 1170.29 | 1206.12 | **+3.06 %** | 0.40 / 0.35 % | SAME / SAME |
| 8B 4w | 541.23 | 554.41 | **+2.44 %** | 0.71 / 0.11 % | SAME / SAME |
| 8B 8da4w (repeat, 21:46 PDT, start 44 C) | 548.47 | 561.56 | **+2.39 %** | 0.56 / 0.36 % | SAME / SAME |

Geomean of the six cells **+4.10 %**, each outside the +-2 % band and far above the A/A floor (+-0.23 %).
The parent arm agrees with `t3-aa` within 0.5 % (2828.73 / 2828.73, 1192.08 / 1172.97, 544.10 / 549.50).
The 8B 8da4w cell is from the repeat (`sessions/c7-softmax-r3/repeat-8b8da4w/`, both arms interleaved, 5 valid
runs each, cool start); in the first session 8 of its runs carry `other_gpu_process` (below) and its 4 valid runs
per arm read 548.0 to 549.1 against 562.0 to 562.5 tok/s, the same gain.

Gate items (all with the candidate's environment):

| item | result |
|---|---|
| `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` | correctness rc = 0; 12 of 12 production-diff cases ALL PASSED; default vs tiled SAME on the real-text and the unaligned prompt; decode 31 tokens; `linear <scheme> rc=1` as on the parent. Status lines identical to the parent's (`s8-r3final/verify.out`) |
| next token parent vs candidate | SAME in all six cells on both prompts |
| SDPA tiers, 12 passes each | `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4 in every pass, 0 mismatches, `pairing=ok` in every case; control with the table kernels 1 pass each, same |
| traces (one warm run per arm) | softmax 115.1 -> 70.5 ms (1B), 150.2 -> 93.3 ms (3B), 229.1 -> 142.3 ms (8B); QK^T and attn*V unchanged |

Where the time is after candidate 7 (`sessions/c7-softmax-r3/trace/families.csv`, candidate arm, ms; 4w / 8da4w):

| family | 1B | 3B | 8B |
|---|---|---|---|
| linear GEMM | 373 / 351 | 1081 / 1030 | 2799 / 2625 |
| softmax | 70 / 70 | 93 / 93 | 142 / 142 |
| QK^T | 47 / 46 | 129 / 127 | 191 / 187 |
| attn*V | 42 / 42 | 77 / 78 | 119 / 119 |
| elementwise (`mul`, `sigmoid`, `add`; upstream) | 68 / 68 | 129 / 130 | 244 / 243 |
| copy / view / other (upstream) | 61 / 36 | 135 / 86 | 229 / 137 |
| 8-bit activation quantize (upstream) | - / 43 | - / 101 | - / 167 |
| RMSNorm + RoPE (upstream) | 14 / 14 | 40 / 41 | 60 / 61 |
| total dispatch | 680 / 675 | 1694 / 1695 | 3797 / 3693 |

Single traced runs: the linear rows of the two arms differ by up to 3 % (8B 4w 2716 against 2799 ms) without any
linear kernel having changed, so differences of that size between sessions are not resolved by a trace.

Why the cell was repeated: in the first session its runs 4 to 8 (eight runs) are marked `other_gpu_process`. The process the session saw was pid 576199,
`/bin/bash`: an interactive monitoring shell of this campaign whose command line contained the name of the
runner binary as text. `e2e5.sh` looks for other GPU processes with `pgrep -af`, which matches full command lines,
so the shell was counted for the 6 minutes it was alive. No GPU process ran (the lock holder was the session; the
flagged runs read 548.0 to 549.4 and 561.9 to 562.5 tok/s, the same as the valid ones). The flag is not removed
and the tool is not changed; the cell is repeated instead. Pitfall for anyone watching a session: do not put the
runner or microbench binary names in a shell command while `e2e5.sh` is running.

Bitwise comparison against the parent (`results/780m/softmax/bitwise-c7.txt`): the raw fp16 SDPA output of
every correctness case of the tiers `all`, `extended`, `peaked` and `full` (21 cases, up to 16.8 MB each, the three
production head configurations at S = 2048 among them) is **byte-identical** between the parent kernels and the
parent kernels with softmax r3. Candidate 7 therefore does not use the near-tie or the reference-error rule.

## Candidate 8 (Part 2): fused SDPA kernel `sarc_dev_780m_sdpa_fused3`: ACCEPTED (reference-error rule, owner decision 2026-10-04)

After candidate 7, QK^T + softmax + attn*V are still 160 of 680 ms on 1B, 299 of 1694 ms on 3B and 452 of
3797 ms on 8B, and all three kernels are bound by the traffic of the S x S attention matrix (QK^T writes it,
the softmax reads and rewrites it, attn*V reads it; about 550 MB per layer on 1B by count of bytes), not by
arithmetic. The fused kernel never writes that matrix. For a block of query rows of one head it walks the context
in blocks, twice: pass A computes the scores (fp32 accumulate, scaled, rounded to fp16, exactly as the QK^T kernel
does) and keeps the row maxima; pass B computes the scores again, e = exp(score - max) in fp16, the row sums in
fp32 and `acc += e V` in fp32; the output is `acc / sum`.

### Gate (session `c8-fused3`, `results/780m/sessions/c8-fused3/`)

Started 21:59 PDT at 44 C. Both arms are the same binary (build `fused5` = the branch + both hook patches in a
scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3` (= candidate 7); the candidate
arm adds `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`. Tok/s, median of 5
valid runs per arm, arms interleaved; recomputed from `runs.csv`:

| cell | parent (candidate 7) | candidate 8 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3020.65 | 3624.78 | **+20.00 %** | 0.15 / 0.18 % | SAME / SAME |
| 1B 8da4w | 3011.76 | 3618.37 | **+20.14 %** | 0.15 / 0.18 % | SAME / SAME |
| 3B 4w | 1232.25 | 1376.34 | **+11.69 %** | 0.06 / 0.20 % | SAME / SAME |
| 3B 8da4w | 1201.88 | 1344.71 | **+11.88 %** | 1.24 / 0.79 % | SAME / SAME |
| 8B 4w | 556.07 | 601.12 | **+8.10 %** | 0.46 / 0.59 % | SAME / SAME |
| 8B 8da4w | 561.71 | 605.74 | **+7.84 %** | 0.19 / 0.21 % | **DIFFER** / SAME |

Geomean **+13.16 %**; 60 timed runs, none rejected; median clock 2783 to 2800 MHz; start temperature 44 to 50 C,
peak 93 C. The unaligned prompt (1972 tokens) is not served by the fused kernel, so its checks compare the same
kernels in both arms.

The one differing item, 8B 8da4w on `prompt_2048.txt` (2048 times the token " the"): the model is undecided there.
Next-token logits of the top candidates in the four arms (`results/780m/probe/c8-differ-8b-8da4w-prompt_2048.csv`):

| token | parent tiled | parent default | candidate tiled | candidate default |
|---|---:|---:|---:|---:|
| 118 (a byte token) | 11.461 (p 0.0126) | 11.500 (p 0.0129) | 11.125 | 11.156 |
| 53 `V` | 11.297 | 11.234 (p 0.0099) | 11.422 (p 0.0119) | 11.367 (p 0.0113) |
| 5531 `cal` | 11.047 | 11.180 | 11.102 | 11.016 |
| top-2 margin | 0.164 | 0.266 | 0.297 | 0.211 |

The parent's top token has probability 1.3 % and leads by 0.27 logit (0.16 in its tiled arm); the candidate moves
token 118 by -0.34 and token 53 by +0.13. Largest logit difference over the vocabulary: candidate default against
parent default 0.66, parent tiled against parent default 0.83; KL 8.2e-3 against 4.6e-3 nat.

Record under the second owner decision of 2026-10-04 (arithmetic changes are judged against a reference). No
threshold was chosen here; the two in item 3 are the decision's.

1. **Gate criterion, error against the reference** (`results/780m/sdpa-error/c8-fused3.csv`; fp64 CPU reference
   computed from the fp16 inputs the device saw; parent = candidate 7 kernels, same inputs). Production shapes
   (tier `full`):

   | case | elements | rms parent -> candidate | maximum parent -> candidate |
   |---|---:|---|---|
   | 1B head configuration, S = 2048 | 4,194,304 | 1.968e-5 -> 1.029e-5 (x 0.52) | 8.44e-4 -> 3.86e-4 (x 0.46) |
   | 3B head configuration, S = 2048 | 6,291,456 | 2.000e-5 -> 1.035e-5 (x 0.52) | 9.52e-4 -> 3.57e-4 (x 0.37) |
   | 8B head configuration, S = 2048 | 8,388,608 | 1.979e-5 -> 1.036e-5 (x 0.52) | 1.039e-3 -> 3.69e-4 (x 0.36) |
   | 8B head configuration, S = 1024, `input_pos` 1024 | 4,194,304 | 9.62e-6 -> 4.79e-6 (x 0.50) | 9.51e-5 -> 4.05e-5 (x 0.43) |

   The candidate's rms and maximum error are not larger than the parent's on every production shape: **met**.
   Outside the production shapes: lower in all 8 `extended` cases (rms x 0.51 to 0.53); in the 5 `peaked` cases rms
   x 0.81 to 0.91, maximum lower in 4 and higher in 1 (`peaked_tiny_gqa_s256`, S = 256, 2 heads: 1.74e-3 ->
   1.98e-3).
   Correctness tiers with the candidate's environment, 12 passes each, 0 mismatches in every pass: `all` 4 of 4,
   `extended` 8 of 8, `full` 4 of 4, `peaked` 5 of 5, `fused` 5 of 5; the fused kernel the only SDPA kernel
   dispatched, `pairing=ok`; one control pass of `all`, `extended` and `full` with the table kernels.
2. **Evidence.** The logits at the differing position for the four arms: the table above. The real-text
   comparison (`results/780m/probe/c8-real-text-compare.csv`; 32 prompts of 128 to 1920 tokens cut from six real
   texts by the fixed recipe of `tools/probe_prompts.py`, next-token distribution after each prompt, 24 runs of
   `logits_probe`):

   | cell | pair | top-1 differs | KL mean / max (nat) | largest logit difference | perplexity ratio |
   |---|---|---:|---|---:|---:|
   | 1B 4w | candidate vs parent (default arms) | 0 of 32 | 2.2e-5 / 1.3e-4 | 0.11 | 1.0029 |
   | | parent tiled vs parent default | 0 | 3.9e-5 / 2.6e-4 | 0.11 | 1.0044 |
   | 3B 4w | candidate vs parent | 0 | 1.5e-5 / 1.1e-4 | 0.09 | 0.9993 |
   | | parent tiled vs parent default | 0 | 3.2e-5 / 5.9e-4 | 0.11 | 0.9993 |
   | 8B 4w | candidate vs parent | 0 | 2.1e-5 / 2.0e-4 | 0.07 | 1.0003 |
   | | parent tiled vs parent default | 0 | 2.6e-5 / 2.5e-4 | 0.11 | 1.0014 |
   | 1B 8da4w | candidate vs parent | 2 | 3.3e-2 / 0.39 | 3.69 | 0.951 |
   | | parent tiled vs parent default | 1 | 3.7e-2 / 0.55 | 3.45 | 0.896 |
   | 3B 8da4w | candidate vs parent | 5 | 3.4e-2 / 0.45 | 3.32 | 0.912 |
   | | parent tiled vs parent default | 3 | 3.6e-2 / 0.47 | 2.91 | 0.937 |
   | 8B 8da4w | candidate vs parent | 4 | 2.6e-2 / 0.30 | 2.63 | 0.990 |
   | | parent tiled vs parent default | 4 | 6.3e-2 / 0.58 | 2.81 | 0.902 |

   (Perplexity of the true next token over the 32 prompts, first arm over second; the csv also has the
   candidate-tiled vs candidate-default rows.) In the 4w cells the candidate is closer to its parent than the
   parent's two linear arms are to each other. The 8da4w cells are three orders of magnitude noisier in every
   pair, including the two parent arms (the activations are re-quantized to 8 bit from their own range at every
   linear layer); the candidate against its parent is of the same size as that spread in each of them (top-1
   differences 2 / 5 / 4 against 1 / 3 / 4, mean KL 0.033 / 0.034 / 0.026 against 0.037 / 0.036 / 0.063).
3. **Gross-divergence check** (reject if, in any cell, the mean KL of candidate-default against parent-default
   exceeds 0.5 nat or the top-1 token differs on more than one third of the prompts): largest mean KL 0.034 nat,
   at most 5 of 32 prompts: **passed**.
4. **Next-token items that differ**: one. 8B 8da4w, `prompt_2048.txt`, parent against candidate (the e2e session).
   Every other item is SAME: parent vs candidate in the other 11 cell x prompt checks, and `verify.sh` default vs
   tiled on the real-text and the unaligned prompt.

The rest of the gate: unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate's
environment: correctness rc = 0, 12 of 12 production-diff cases ALL PASSED, default vs tiled SAME on both
prompts, decode 31 tokens, `linear <scheme> rc=1` as on the parent. Traces (one warm run per arm,
`sessions/c8-fused3/trace/`): total dispatch time 672.5 -> 563.9 ms (1B 4w), 1657 -> 1499 ms (3B 4w), 3700 ->
3458 ms (8B 4w); QK^T + softmax + attn*V 158.9 / 296.2 / 449.4 ms are gone and the copy pass + fused kernel add
45.3 / 120.7 / 184.2 ms (they are in the trace tool's "copy/view/other" family); the other families are
unchanged within 2 %. Kernel time per layer at a steady clock after the gate, three runs
(`results/780m/fused/kernel-time-steady-c8-gate.csv`): 9.96 -> 2.82 ms (1B), 10.24 -> 4.32 to 4.36 ms (3B),
13.51 -> 5.54 to 5.66 ms (8B).

### Structures

Three structures were built and measured; the third is the candidate.

| structure | what a workgroup is | kernel time per layer, 1B / 8B head configuration |
|---|---|---|
| three kernels (candidate 7) | | 9.97 / 13.52 ms |
| `fused`: K and V staged in shared memory, 4 to 6 barriers a block | 4 to 8 subgroups, 64 or 128 rows | 3.84 / 11.7 ms |
| `fused2`: staging double-buffered (1 barrier a block), score rows private to a subgroup, V transposed | 2 to 8 subgroups | not faster than `fused` (first 5 runs: 6.2 against 5.9 ms on 1B) |
| `fused3`: one subgroup, no staging, no barrier; Q tiles in registers, K from the cache buffer, V from a transposed copy | 1 subgroup, 16 or 32 rows | 2.55 / 9.95 ms |
| `fused3`, packed (`...rk`): K and V read from tile-packed copies in which a 16 x 16 operand tile is 512 contiguous bytes | same | **2.80 / 5.57 ms, copy pass included** (2.46 / 4.84 ms without it) |

(Steady GPU clock: 40 warm-up and 10 timed runs, median; `results/780m/fused/kernel-time-steady.csv`. With the
default 3 + 5 runs of the microbench the clock is still rising and the fused kernels read up to 1.7 times slower:
`kernel-time-development.csv`, where only values in the same column compare.)

What the measurements say about why:

- The disassembly shows two 128-bit shared-memory loads per lane for every operand tile; `fused` and `fused2` load
  1.0 to 1.5 tiles per multiply-add, which the roofline table above puts at a small fraction of the matrix roof.
  Removing the row padding from the shared tiles made `fused2` 2.6 times slower (15.5 against 5.9 ms): shared
  memory loads were the limit, not the barriers.
- Keeping the Q tiles in registers and reading K straight from the buffer halves the loads per multiply-add.
  Reloading Q from the buffer for every block instead costs 2.2 to 3.0 times (1B 10.2 against 4.6 ms, 8B 44.7
  against 15.0 ms, first-5-run values): loads from the cache buffers are only cheap when a tile is contiguous. In
  the caches a 16 x 16 tile is 16 runs a row stride apart; the packed copies make it one run, which is what
  brought head_dim 128 from 9.95 to 4.96 ms (16 x 16 tile, steady clock).
- The packed `fused3` runs 3.1 M (1B) and 6.3 M (8B) multiply-adds a layer in 2.46 and 4.84 ms: about 71 % of the
  matrix roof. Without pass A the same kernel takes 1.58 / 3.12 ms (measurement-only variant `...rkm1`): a
  one-pass (running-maximum) form would save at most that, about 2 % end to end; not attempted.
- The copy pass (`sarc_dev_780m_sdpa_kvt`, 4 to 16 MB a layer) costs 0.35 (1B) to 0.7 ms (8B), three times what
  the bytes cost at the copy roof; a 16 x 4 workgroup instead of 8 x 8 is slower (2.36 against 2.27 ms on 1B);
  not tuned further.

### One-pass variant (`...rko`, built and measured, a later candidate)

`ONLINE` keeps a running row maximum instead of pass A: when a block raises the maximum of a row, that row's
accumulator and sum are scaled by exp(old - new) (an element-wise divide of the accumulator tiles by a divisor
tile from shared memory, only in subgroups where some row moved). Build `fused6`: correct on tiers `all`,
`extended`, `peaked` and `fused` (one pass each, 0 mismatches), error against the fp64 reference equal to or
slightly below the two-pass kernel's. Kernel time per layer, copy pass included, steady clock: 1B 2.80 -> 2.27 ms,
3B 4.33 -> 3.22 ms, 8B 5.54 -> 4.04 ms. End to end that is about +1.5 to +2 %, so it is not gated alone: it goes
into the next candidate together with the Part 1 winners per shape.

Variants chosen (`ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`): for head_dim
64, 32 rows x 32-column blocks (2.80 ms; 16 x 32 reads 2.77 to 2.79 ms, a tie); for head_dim 128, 16 rows x 64
(8B 5.56 to 5.58 ms, 3B 4.32 ms; 16 x 16, 16 x 32 and 32 x 32 are 1.5 to 4 % behind on both models, twice).

- Files: `glsl/sarc_dev/sarc_dev_780m_sdpa_{fused,fused2,fused3,vt,kvt}.{glsl,yaml}`,
  `impl/sarc_dev/Sdpa780mFused.cpp`, the 780m block of `Overrides.cpp`.
- It needs `hooks/sdpa-fused-hook.patch` (about 70 added lines in the release zone and `SDPA.cpp`, not applied on
  this branch; described in `hooks/README.md`): nodes appended after the three SDPA nodes, and an empty dispatch
  for those three when the fused node serves the call. It serves prefill calls whose S and `input_pos` are
  multiples of the row tile and the context block; decode, unaligned prompts and `ET_VK_DISABLE_COOPMAT` keep the
  three kernels. It also serves aligned lengths the 128-row QK^T tile does not (64, 192, ...), which the stock path
  runs on the tiled kernels.
- It changes the arithmetic (the three kernels round e / sum to fp16 before attn*V; this kernel rounds e and
  divides the fp32 accumulator), so it is judged by the reference-error rule of the second owner decision.
- Test additions (`test/sarc_dev/test_llama_microbench.cpp`, no tolerance and no existing case changed): the fused
  kernel is recognised by the SDPA cases (it must then be the only SDPA kernel dispatched);
  `ET_VK_SDPA_ERROR_REPORT=1` prints rms and maximum error against an fp64 reference computed from the fp16
  inputs; tier `peaked` (Q scaled by 8, so a few context positions carry most of a row's weight; the existing
  tiers have near-uniform attention); tier `fused` (S = 32, 64, 192, 320, shapes only the fused kernel takes);
  `ET_VK_SDPA_PERF_RUNS` for the steady-clock timing. `test/sarc_dev/probe/logits_probe` writes the next-token
  logits of token prompts, for the real-text comparison (`tools/probe_{prompts,run,compare}.*`).

Results so far (one pass each unless said; `<artifacts>/fx/`, `<artifacts>/stage/c8-fused3/`):

- Correctness, chosen variants: tiers `all` 4 of 4, `extended` 8 of 8, `peaked` 5 of 5; tier `fused` 12 passes,
  5 of 5 each; 0 mismatches, the fused kernel the only SDPA kernel dispatched.
- Error against the fp64 reference, three kernels -> fused: rms 1.9e-5 to 4.6e-5 -> 0.95e-5 to 2.4e-5 on the 8
  `extended` cases (about half), maximum lower in all 8; on the 5 `peaked` cases rms 2.6e-4 to 2.7e-4 -> 2.2e-4 to
  2.4e-4, maximum lower in 4 and higher in 1 (`peaked_tiny_gqa_s256`: 1.74e-3 -> 1.98e-3; both arms round the
  same fp16 scores there). The `full` tier, which holds the production shapes the rule names, runs in the gate.
- End to end, one run per arm, not a session: 1B 4w 3020.65 -> 3624.78 tok/s (+20.0 %). Projected from the
  kernel times and the candidate-7 traces: about +20 % (1B), +12 % (3B), +8 % (8B).

## Candidate 9: candidate 8 + one-pass fused kernel + 4w kernel per shape: gate passed, +3.19 % geomean

Session `c9-online-q4` (`results/780m/sessions/c9-online-q4/`), started 02:18 PDT at 44 C. Both arms are the same
binary (build `fused7` = the branch + both hook patches) with candidate 8's environment; the candidate arm uses
the one-pass variants (`ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko`) and
`ET_VK_SARC_780M_PROFILE=refine9`. Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (candidate 8) | candidate 9 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3631.21 | 3806.69 | **+4.83 %** | 0.35 / 0.19 % | SAME / SAME |
| 1B 8da4w | 3624.78 | 3690.09 | +1.80 % | 0.18 / 0.00 % | SAME / SAME |
| 3B 4w | 1377.27 | 1439.21 | **+4.50 %** | 0.14 / 0.14 % | SAME / SAME |
| 3B 8da4w | 1348.26 | 1362.61 | +1.06 % | 1.24 / 1.21 % | SAME / SAME |
| 8B 4w | 601.29 | 634.06 | **+5.45 %** | 0.18 / 0.43 % | SAME / SAME |
| 8B 8da4w | 606.10 | 615.75 | +1.59 % | 0.33 / 0.33 % | SAME / SAME |

Geomean **+3.19 %**; 60 timed runs, none rejected; clock 2800 MHz. The 8da4w cells only have the one-pass fused
kernel (their linear kernel is unchanged) and are inside the +-2 % band: by the campaign's rule that part is not
a gain by itself. The 4w cells have both changes and are outside the band.

The two parts, measured separately before the gate:

- 4w kernels (`refine9`): the raw output of all 12 production-diff cases is byte-identical to `780m-refine3`
  (`results/780m/space/bitwise-4w-refine9.txt`), so this part does not change the arithmetic.
- One-pass fused kernel: it does (rescaling instead of a first pass), so it is judged against the reference. On
  the four production shapes its rms error is 0.996 to 0.998 times candidate 8's and its maximum is equal or lower
  (`results/780m/sdpa-error/c9-online-full-precheck.csv`); the gate repeats this.

Gate (finished 04:10 PDT), all with the candidate's environment:

| item | result |
|---|---|
| `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` | correctness rc = 0; 12 of 12 production-diff cases ALL PASSED; default vs tiled SAME on the real-text and the unaligned prompt; decode 31 tokens; `linear <scheme> rc=1` as on the parent; status lines identical to candidate 8's. The 4w texture3d cases dispatch the `refine9` kernels |
| next token parent vs candidate | SAME in all six cells on both prompts: **no differing item** |
| SDPA tiers, 12 passes each | `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4, `peaked` 5 of 5, `fused` 5 of 5 in every pass, 0 mismatches, the fused kernel the only SDPA kernel dispatched, `pairing=ok`; controls with the table kernels 1 pass each |
| error against the fp64 reference (`results/780m/sdpa-error/c9-online-q4.csv`) | production shapes: rms 0.996 to 0.998 times candidate 8's, maximum equal or lower: **not larger on any of them**. All 17 cases of `extended`, `peaked` and `full`: rms ratio 0.992 to 1.000 |
| real-text comparison, 32 prompts, six cells (`results/780m/probe/c9-real-text-compare.csv`) | 4w cells: top-1 differences 0, mean KL 1.6e-5 to 2.2e-5 (candidate 8 tiled vs default: 1.9e-5 to 3.9e-5). 8da4w cells: top-1 differences 3 / 4 / 1 of 32, mean KL 0.028 / 0.020 / 0.017 (candidate 8 tiled vs default: 3 / 3 / 2 and 0.045 / 0.027 / 0.027). Gross-divergence check passed (limits 0.5 nat, one third of the prompts) |
| traces, one warm run per arm (`sessions/c9-online-q4/trace/`) | 4w linear GEMM 367.5 -> 350.9 ms (1B, -4.5 %), 1049 -> 1018 ms (3B, -2.9 %), 2659 -> 2551 ms (8B, -4.1 %); 8da4w GEMM unchanged (+0.1 / +0.1 / -0.5 %); copy pass + fused kernel -10 / -31 / -48 ms in both schemes; totals 560.0 -> 533.3, 1481 -> 1420, 3382 -> 3228 ms (4w) |
| kernel time per layer, steady clock, three runs (`results/780m/fused/kernel-time-steady-c9-gate.csv`) | copy pass + fused kernel 2.82 -> 2.26 ms (1B), 4.31 -> 3.22 ms (3B), 5.54 -> 4.05 ms (8B) |

The candidate changes the SDPA arithmetic but no next-token item differs, so nothing has to be excused; the
reference-error evidence is recorded anyway. The 8da4w cells' gain (+1.06 to +1.80 %) comes from the one-pass
kernel alone and is inside the band; whether that kernel is kept for 8da4w is therefore a matter of preference
for the simpler two-pass form, not of a measured gain.

## Part 1, 4w: the best kernel per shape within the existing bodies (`results/780m/space/confirm-4w/`)

Search: 2,444 of 2,500 random survivors screened, 991 neighbours of the 22 best screened, then the 95
configurations within 8 % of the fastest on some screened shape measured in full (all twelve shapes, 3 + 5 runs),
and the 10 fastest per shape (35 configurations) five times. Repeat spread of the leaders 0.1 to 0.4 %.
Kernel time relative to the fastest of the shape (`summary.csv`):

| shape (N, K) | fastest | `780m-refine3` choice | shipped `t128x128..c` | `t128x256..cbt` | `t256x128..g18..bbt` | `..g24..bbt` | `..g28..bbt` |
|---|---|---:|---:|---:|---:|---:|---:|
| 1B wq_wo (2048, 2048) | `t128x256k32g42s32f32cbt` 1586 us | +2.9 % | +2.6 % | **0** | +5.6 % | +3.2 % | +0.4 % |
| 1B wk_wv (512, 2048) | `t128x128k32g42s32f32xpiiw` 439 us | +0.3 % | **+0.3 %** | +5.1 % | +9.3 % | +9.9 % | +5.2 % |
| 1B w1_w3 (8192, 2048) | `t256x128k32g28s32f32bbt` 6019 us | +4.6 % | +6.1 % | +3.2 % | +9.1 % | +5.8 % | **0** |
| 1B w2 (2048, 8192) | `t256x128k32g18s32f32bbt` 5887 us | +7.5 % | +11.1 % | +5.4 % | **0** | +6.4 % | +5.8 % |
| 3B wq_wo (3072, 3072) | `t256x128k32g24s32f32bbt` 3410 us | +3.9 % | +5.2 % | **+1.4 %** | +2.0 % | 0 | +2.6 % |
| 3B wk_wv (1024, 3072) | `t128x256k32g42s32f32cbt` 1203 us | +2.7 % | +5.1 % | **0** | +7.9 % | +4.7 % | +3.9 % |
| 3B w1_w3 (8192, 3072) | `t256x128k32g24s32f32bbt` 9136 us | +3.3 % | +4.5 % | **+1.5 %** | +1.9 % | 0 | +4.2 % |
| 3B w2 (3072, 8192) | `t256x128k32g18s32f32bbt` 8596 us | +6.9 % | +11.7 % | +4.5 % | **0** | +3.4 % | +3.0 % |
| 8B wq_wo (4096, 4096) | `t256x128k32g18s32f32bbt` 5980 us | +3.6 % | +9.4 % | +1.1 % | **0** | +1.3 % | +4.3 % |
| 8B wk_wv (1024, 4096) | `t128x256k32g42s32f32cbt` 1590 us | +1.7 % | +8.2 % | **0** | +1.8 % | +2.0 % | +3.4 % |
| 8B w1_w3 (14336, 4096) | `t256x128k32g18s32f32bbt` 20766 us | +4.5 % | +7.5 % | +2.0 % | **0** | +1.4 % | +4.3 % |
| 8B w2 (4096, 14336) | `t256x128k32g18s32f32bbt` 20408 us | +5.3 % | +9.7 % | +2.5 % | **0** | +2.3 % | +0.9 % |

Bold = what profile `refine9` runs. Its rule (`Overrides.cpp`, 780m block): N below 1024 keeps the shipped tile;
N >= 2048 with K >= 4096 takes the 256-row tile `t256x128k32g18s32f32bbt`; N >= 8192 with K below 3072 takes
its 16-subgroup grid `..g28..`; everything else from N = 1024 takes `t128x256k32g42s32f32cbt`. Where two kernels
are within 2 % the rule keeps the one it already uses (3B wq_wo and w1_w3: `..cbt` instead of `..g24..`).
Per layer (2 wq_wo + 2 wk_wv + 2 w1_w3 + w2) against `780m-refine3`: best per shape -4.76 / -4.03 / -4.26 %
(1B / 3B / 8B); `refine9` -4.75 / -3.05 / -4.26 %.

What the confirmed set says about the parameters near the optimum (`local-effects.csv`: pairs of confirmed
configurations that differ in one parameter, full measurement, ratio of kernel times over all shapes):

| parameter | change | median | range | reading |
|---|---|---:|---|---|
| `IMG_A` | off -> on | 1.000 | 0.997 to 1.009 | does not matter |
| `IMG_W` | off -> on | 1.000 | 0.987 to 1.009 | does not matter |
| drain mode | band / pooled full / in Ash | 0.993 to 1.003 | 0.976 to 1.030 | does not matter (the three usable modes are within 3 %) |
| texel-wise staging | bx -> plain | 1.003 | 0.866 to 1.180 | no effect on average; interacts with the tile (-13 to +18 %) |
| `B_COLMAJOR` | off -> on | 0.993 | 0.815 to 1.047 | decides the 256-row tile (up to -18 %), neutral to +5 % elsewhere |
| `FRAG_LAYOUT` | off -> on | 0.968 | 0.958 to 0.986 | -1 to -4 % on the 12 pairs present; it cannot be combined with `B_COLMAJOR`, and those pairs are 5 % or more behind the leaders |
| `SH_F16V4` | off -> on | 1.028 | 1.023 to 1.035 | always slower |
| tile M | 128 -> 256 | 1.013 | 0.862 to 1.188 | by shape: better for large K, worse for small N |
| tile N | 128 -> 256 | 0.970 | 0.833 to 1.080 | by shape, with M and the grid |
| grid X / Y | 2 -> 4, 1 -> 2, 4 -> 8 | 0.985 to 1.012 | 0.917 to 1.108 | by shape |

Beyond the parameter space, tried on the winners and not pursued: a slab layout of both operands in shared
memory (`tools/gen_sl.py`: A as `FRAG_LAYOUT` has it, B column-major as [k16 slab][n][16 k], no padding; the
layout the 8da4w zpg kernel uses). It is 8 to 25 % slower than the padded column-major kernels of the same tile on
all twelve shapes (`results/780m/space/slab/kernel-time.txt`); the generated files are not kept in the tree.

Tile K (32), subgroup size (32) and the fp32 accumulator have no alternative within 14 % of the leaders
(refinement round 1).

After candidate 9 the local search was continued until it stopped moving (`results/780m/space/{refine2,refine3,
geo-cbt}/`, full measurement on all twelve shapes):

- Round 2, the 51 untested single-parameter neighbours of the winners above: the in-Ash drain instead of the
  band drain on the 256-row tiles (`t256x128k32g{18,24,28}s32f32cbt`) is 0.3 to 2.3 % faster on 9 of the 12
  shapes; nothing else moves.
- Round 3, the 29 untested neighbours of those: `t128x128k32g24s32f32cbt` (a 2 x 4 grid on the shipped tile
  size, column-major B) is the fastest on the two small 1B shapes (wk_wv 420 against 440 us, wq_wo 1544 against
  1586 us); `IMG_A` / `IMG_W` on the leaders are within 0.3 %.
- Geometry scan: the 85 surviving geometries with the leading flag set (fp32 accumulator, in-Ash drain,
  column-major B; subgroup 32, tile K 16 or 32, M and N at least 64, 2 to 32 MMA tiles per subgroup) that no
  earlier stage had measured: none enters the first five on any shape.

Second confirmation (`results/780m/space/confirm2-4w/`): the 28 leaders of all of the above, 5 full
measurements each, repeat spread 0.15 to 1.1 %. **Within the existing kernel bodies the best 4w kernel per shape
on this device and driver is:**

| shapes | kernel (profile `refine10`) | tied within 2 % | against `780m-refine3` | against `refine9` |
|---|---|---|---:|---:|
| K >= 4096, N >= 1024 (all of 8B; w2 of 1B and 3B) | `t256x128k32g18s32f32cbt` | its `IMG_A` / `IMG_W` twins; for the three w2 shapes and 8B wk_wv also the band-drain twins | -2.2 to -7.2 % | -0.1 to -2.9 % |
| K = 3072, N >= 2048 (3B wq_wo, w1_w3) | `t256x128k32g24s32f32cbt` | `t256x128k32g18s32f32cbt` | -4.8 to -5.6 % | -3.1 to -3.4 % |
| K = 3072, N = 1024 (3B wk_wv) | `t128x256k32g42s32f32cbt` | ten flag variants of the same tile | -2.6 % | 0 |
| K = 2048, N = 8192 (1B w1_w3) | `t256x128k32g28s32f32cbt` | its `IMG` twins and band-drain twins | -6.1 % | -1.7 % |
| K = 2048, N <= 2048 (1B wq_wo, wk_wv) | `t128x128k32g24s32f32cbt` | wq_wo: `t256x128k32g28s32f32cbt`; wk_wv: none (the shipped tile is 4.5 % behind) | -5.4 / -4.6 % | -2.7 / -4.6 % |

Per layer: -6.10 / -5.30 / -5.93 % linear time against `780m-refine3` (1B / 3B / 8B) and -1.41 / -2.37 / -1.82 %
against `refine9`. The raw output of the 12 production-diff cases is byte-identical to `780m-refine3`
(`results/780m/space/bitwise-4w-refine10.txt`). Still open: the 12 production-diff passes per confirmed
configuration (3 done, all passed).

## Candidate 10: candidate 9 with the refined 4w table (`refine10`): gate passed, +0.47 %, inside the band

Session `c10-q4-refine10` (`results/780m/sessions/c10-q4-refine10/`), started 08:19 PDT at 43 C; same binary
(build `fused9`) in both arms, candidate 9's environment with `ET_VK_SARC_780M_PROFILE=refine10` instead of
`refine9`:

| cell | parent (candidate 9) | candidate 10 | gain | repeat spread parent / candidate | next token |
|---|---:|---:|---:|---|---|
| 1B 4w | 3806.69 | 3842.40 | +0.94 % | 0.00 / 0.38 % | SAME / SAME |
| 1B 8da4w | 3690.09 | 3690.09 | 0.00 % | 0.36 / 0.18 % | SAME / SAME |
| 3B 4w | 1439.21 | 1459.73 | +1.43 % | 0.14 / 0.00 % | SAME / SAME |
| 3B 8da4w | 1370.82 | 1360.80 | -0.73 % | 1.47 / 1.41 % | SAME / SAME |
| 8B 4w | 634.84 | 642.21 | +1.16 % | 0.15 / 0.41 % | SAME / SAME |
| 8B 8da4w | 614.83 | 615.20 | +0.06 % | 0.30 / 0.27 % | SAME / SAME |

Geomean **+0.47 %**, every cell inside the +-2 % band: by the campaign's rule not a gain, and the first of the
two candidates under 2 % that end the campaign. `verify.sh`: correctness rc = 0, 12 of 12 production-diff cases
ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens, `linear <scheme> rc=1` as on the parent; no
SDPA kernel changes, so the SDPA tiers were not repeated. Traces: 4w linear GEMM 350.6 -> 346.8 ms (1B, -1.1 %),
1015.9 -> 995.4 ms (3B, -2.0 %), 2538.4 -> 2508.4 ms (8B, -1.2 %). The 4w cells move by what the kernel timing predicts (+0.9 /
+1.7 / +1.4 %); the 8da4w cells run the same kernels in both arms and scatter by -0.7 to +0.1 %.

## Can the sweep slot in between two timed runs of a session? No (checked 2026-10-04 18:44 PDT, during `c7`)

Asked by the owner after seeing `sweep_space.py` alive and the GPU at 79 C during the candidate-7 session.

- `e2e5.sh` takes the gpu-lab lock once, at the start of the script (`exec 9>>lock; flock 9`), and holds it until
  it exits: the lock is per session, not per run. `lslocks` during the session shows `e2e5.sh` (pid 565053) as the
  holder of `~/.cache/gpu-lab/lock-00000000-c400-...`. `sweep_space.py` takes the same lock for each of its own
  runs, so it cannot run while a session holds it.
- Independently of the lock, the sweep waits while `<artifacts>/PAUSE` exists; `gate2.sh` creates it before the
  cool-down wait and removes it when the whole gate is done. `PAUSE` has existed since 17:42 PDT.
- The enumeration process seen alive (pid 537928) is that paused sweep. Its last result row is stamped
  21:15:48 UTC (14:15 PDT), four hours before the session's first run (01:13:58 UTC); `raw/space/results.csv` and
  `sweep.out` have not been written since. No microbench process ran during the session.
- 79 C is the in-run temperature of `llama_main` itself (the runs of this session end at 60 to 73 C and peak
  higher; the earlier sessions peaked at 88 to 95 C). The session started at 45 C after the full 30 min wait
  (`prestart.txt`: the device did not get below 45 C this evening; `t3-aa` started at 42 C).

Conclusion: the session is not affected and is not repeated. The one way a sweep job could run next to a
session would be a session script that locks per run; none of the session tools does.

## Owner decisions in force (read from `CAMPAIGN.md`, section "Owner decisions", 2026-10-04)

- Next-token near-ties: a `DIFFER` no longer rejects by itself if the logits of all four arms, a real-text
  comparison on at least 32 prompts and the parent-tiled vs parent-default floor are measured; recorded as
  `ACCEPTED (near-tie, owner decision 2026-10-04)`. Not used so far: no `DIFFER` has occurred in this round.
- Arithmetic changes are judged against the fp32 reference (rms and maximum error not larger than the parent's
  on the production shapes), with a gross-divergence check; recorded as `ACCEPTED (reference-error rule, ...)`.
  A change meant to be bit-identical does not use this rule and must show bit-identical output. Candidate 7
  (softmax r3) is meant to be bit-identical on every element that is read; the bitwise comparison of the SDPA
  output against the parent is still owed. Candidate 8 (fused SDPA) changes the arithmetic and uses this rule.
- Large parameter spaces: the method below.

## Owner decision, 2026-10-04 (relayed): how the 4w family is searched

The full 4w space (26.8 days) is not wanted and the staged plan (geometries first, flags afterwards) is dropped
before any of its rows were measured. Instead:

1. cost per configuration broken down, and a cheap screening mode validated against the full measurement by rank
   correlation on at least 50 configurations;
2. a uniform random sample of 2,500 of the 125,712 survivors, seed 20261004
   (`tools/sample_space.py`; survivor list sha256 `5b0cd644...605d75`, `enum_space.py --list 4w`), screened;
3. parameter importance and pair interactions from that sample, and a check of the staged plan's independence
   assumption against it;
4. local refinement around the best 20 sampled configurations (one or two parameters at a time);
5. the best 10 per shape get the full measurement (5 repeats, 12 correctness passes).

8da4w, QK^T and attn*V stay full enumerations.

### Cost of one 4w configuration (`raw/cost/cost-1.csv`, one configuration, texture3d, correctness off)

| mode | shapes | runs per shape | wall |
|---|---:|---|---:|
| no case (process start, Vulkan device, exit) | 0 | | 0.05 s |
| full measurement | 12 | 3 warm-up + 5 timed | 16.3 s |
| setup only | 12 | 0 + 1 | 12.1 s |
| `w1_w3` of each model | 3 | 3 + 5 | 5.3 s |
| `w1_w3` of each model | 3 | 1 + 2 | 4.6 s |
| `wq_wo` of each model | 3 | 1 + 2 | 2.2 s |

About 74 % of the full measurement is per-shape setup in the test harness (host-side generation, packing and
upload of up to 58.7 million 4-bit weights per shape, graph build, pipeline creation): 0.26 to 2.2 s per shape,
growing with N x K. The 84 extra runs cost 4.2 s. Process start is negligible and one shader module is compiled
per process, so neither is worth optimising. A run without warm-up reads 1.5 to 2.2 times too slow, so the
screening mode keeps one warm-up run.

### Screening mode against the full measurement (`results/780m/space/validate/`)

The first 64 configurations of the random sample (a prefix of the draw, so itself a uniform sample), each
measured in full (12 shapes, 3 + 5 runs; 24.6 s a configuration on this sample) and in five candidate modes.
Spearman rank correlation of the mode's kernel time with the model's linear time per layer from the full
measurement (2 `wq_wo` + 2 `wk_wv` + 2 `w1_w3` + `w2`), per model (8B / 1B / 3B), and how many of the full top 10
are in the mode's top 10 and top 20:

| mode | s per configuration | rho, 8B / 1B / 3B | top 10 in top 10 | top 10 in top 20 |
|---|---:|---|---|---|
| `wq_wo`, 1 + 1 runs | 2.4 | 0.996 / 0.995 / 0.990 | 10 / 10 / 9 | 10 / 10 / 10 |
| **`wq_wo`, 1 + 2 runs** | 2.7 | 0.995 / 0.996 / 0.997 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wq_wo`, 1 + 3 runs | 3.1 | 0.995 / 0.997 / 0.997 | 10 / 10 / 9 | 10 / 10 / 10 |
| `w1_w3`, 1 + 2 runs | 6.5 | 0.995 / 0.998 / 0.998 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wk_wv`, 1 + 2 runs | 1.5 | 0.992 / 0.992 / 0.996 | 10 / 10 / 10 | 10 / 10 / 10 |

Against every single shape of the full measurement the screening shape has rho 0.972 to 1.000 (lowest for `w2`,
the K = 8192 / 14336 shape). All 64 configurations dispatched their own kernel on all 12 shapes; the kernel-time
coefficient of variation over the 5 timed runs has median 0.5 % and 90th percentile 2.2 %.

Chosen: `--op=wq_wo --runs=1,2` (N = K = 2048 / 3072 / 4096). `wk_wv` is cheaper and ranks as well here, but its
N = 512 / 1024 is the one shape where the wide tiles were measured to lose in round 1, so it is not used alone.
What the agreement does not show: the sample spans a 10x range of kernel times, so a high rho is expected; the
top-10 overlap is the sharper check, and the confirmation stage measures the leaders on all 12 shapes anyway.


## Done in this round

| step | session | result |
|---|---|---|
| baseline re-check, cool start (40 C) | `t1-recheck` | `dev/1.5` 2698.29 / 2544.10 (1B 4w / 8da4w), 1147.98 / 1043.30 (3B), 523.65 / 488.32 (8B) tok/s: within 0.8 % of `sarc-1.5-e2e-benchmark/results/cells.csv` (2698.29 / 2544.1, 1151.21 / 1051.33, 525.80 / 489.13). `780m-refine3` over `dev/1.5`: +4.98 / +11.19, +3.84 / +11.98, +3.99 / +12.44 %, geomean +8.00 %. The flagged 8B 4w cell, re-measured cool: **+3.99 %** (523.65 -> 544.54). Binaries of 2026-10-03, not rebuilt. |
| A/A, warm start | `t2-aa` (superseded) | started 90 s after the re-check (idle reference 44 C, run starts at 49 C): a warm-up, not used as the noise floor. Geomean +0.01 %, cells within +-0.14 %. |
| A/A, cool start (42 C, the idle temperature after 30 min) | `t3-aa` | **noise floor**: geomean +0.01 %, cells -0.23 to +0.19 %, repeat spread at most 0.52 %, clock 2797 to 2800 MHz, next token SAME in all cells on both prompts. Per cell (tok/s, 4w / 8da4w): 1B 2828.73 / 2828.73, 3B 1192.08 / 1172.97, 8B 544.10 / 549.50. |

### The random sample: 2,500 4w configurations screened (`results/780m/space/rand/`)

Seed 20261004, `wq_wo` of each model with 1 + 2 runs, 15:29 to 17:40 PDT. After the first 136 configurations the
cooling limit between runs was raised from 55 to 62 C (it cost 4.5 s a configuration); the 64 validation
configurations re-screened at 62 C differ from their 55 C values by a median of +0.08 % (5th to 95th percentile
-2.9 to +3.3 %, which is the repeatability of this mode). 2,444 configurations were timed; no process failed and
no Vulkan context was lost. The other 56 (2.2 %) kept the table kernel on all shapes: the selector's shared-memory
model (`impl/sarc/Select.cpp`, release zone) refuses them although the shader fits (it does not know the band
drain or `FRAG_LAYOUT`); reaching them needs a hook.

The space is mostly bad and sharply peaked: the median configuration is 2.23x slower than the sample's fastest,
the slowest 248x; 0.1 % are within 2 % of the fastest, 0.3 % within 10 %, 10 % within 1.5x.

Fastest of the sample (geomean kernel time of the three `wq_wo` shapes): `bx_t128x256k32g42s32f32xpiw` 3612 us,
`t128x128k32g42s32f32ciw` 3659 us (the shipped tile + `IMG_W`), `bx_t128x64k16g22s32f32xpiw` 3687 us,
`t256x128k32g42s32f32bbt` 3718 us. The shipped tile and the refine3 wide tile themselves are measured in the same
mode in refinement round 1 (they are not in the sample).

Parameter importance (`importance/importance.csv`, `levels.csv`; preliminary, the local effects around the optimum
come from the refinement). "Range" is between the geomean times of the best and the worst level over the whole
sample; "unique R^2" is what the additive model of log time loses without the parameter; "best-of-level spread"
is how much slower the fastest sampled configuration of the worst level is than the fastest of the best level
(what the parameter still costs when everything else is chosen well; it is a minimum over a sample, so about 3 %
of it is noise):

| parameter | range over the sample | unique R^2 | best-of-level spread | fastest level / levels of the fastest 5 % |
|---|---:|---:|---:|---|
| `WG_TILE_M` | 380 % | 0.254 | 47 % | 128 (63 % of the fastest 5 %), then 64, 256; 32 never |
| `WG_TILE_N` | 272 % | 0.154 | 35 % | 128, 64, 256 all present; 32 never |
| `SG_GRID_Y` | 118 % | 0.108 | 23 % | 2, then 4, 1 |
| `SG_GRID_X` | 29 % | 0.100 | 16 % | 2 and 4 |
| ACC (accumulator) | 129 % | 0.080 | 52 % | fp32: 90 % of the fastest 5 %, best in all 21 geometry strata |
| `WG_TILE_K` | 144 % | 0.028 | 27 % | 32 and 16; 64 is 27 % behind at best |
| `SUBGROUP_SIZE` | 0.6 % | 0.013 | 19 % | 32 (84 % of the fastest 5 %): no effect on average, large near the optimum |
| CSH (drain) | 46 % | 0.002 | 38 % (full without pool) | in Ash, band and pooled full within 3 % of each other |
| `IMG_A` | 4.2 % | 0.000 | 6.9 % | off |
| `SH_F16V4` | 2.3 % | 0.000 | 6.5 % | off |
| `FRAG_LAYOUT` | 6.0 % | 0.000 | 4.4 % | off |
| `B_COLMAJOR` | 9.5 % | 0.000 | 2.9 % | either |
| `IMG_W` | 4.2 % | 0.000 | 2.9 % | either |
| texel-wise staging (`bx`) | 1.2 % | 0.000 | 1.3 % | either: does not matter |
| derived: MMAs per subgroup, (M/Y/16) x (N/X/16) | 4485 % | (eta^2 0.605 alone) | | 8 and 16 (4 to 32 usable); 64 and more are 11 to 25x slower on average |
| derived: threads per workgroup | 340 % | (eta^2 0.062) | 46 % | 256, then 128 |

One derived quantity explains more of the variance (60 %) than all 14 yaml parameters additively (63 %): how
many 16 x 16 MMA tiles one subgroup owns. That is why M, N and the grid look important and interact.

Pair interactions (`importance/pairs.csv`; gain in R^2 over the additive model, noise level about 0.001):
tile M x ACC 0.038, tile N x ACC 0.034, tile K x ACC 0.032, M x grid Y 0.021, M x N 0.019, M x K 0.016,
N x K 0.014, K x CSH 0.007, N x grid X 0.006. No pair of two boolean options is above 0.003.

The staged plan's independence assumption, checked (`importance/independence.txt`): it does not hold. The
additive model explains 63.1 % of the variance of log time; geometry x geometry pairs add 14.4 points,
geometry x option pairs add 12.3 points, option x option pairs 1.2 points (all pairs: 87.9 %). The options do not
interact with each other, but they do interact with the tile geometry about as strongly as the geometry
parameters interact among themselves. For the accumulator the interaction changes the size of the effect, not
its sign (fp32 is best in every stratum). For the drain mode and the boolean options the best level changes
with the geometry (drain: a different best level in 10 of 21 strata; `FRAG_LAYOUT` 5, `IMG_A` 5, `IMG_W` 8), so
"best geometry first, then best flags on it" could have missed combinations; at the size of those effects
(1 to 10 %) that is exactly the margin this campaign is looking for.

### Refinement round 1 (`results/780m/space/refine1/`, 20:21 to 21:41 PDT, screening mode)

1,002 configurations that differ from one of the 22 best sampled ones in one parameter (and in two for the best
five); 991 dispatched their kernel on the three screened shapes. `tools/refine_summary.py`:

- Fastest by geomean of the three shapes: `t256x128k32g24s32f32bbt` 3533 us, `t256x128k32g18s32f32bbt` 3535,
  `bx_t128x256k32g42s32f32xpi` 3578, `t128x256k32g42s32f32cbt` 3598; the sample's best was 3612 and the
  refine3 tile `t128x256k32g42s32f32c` reads 3623 (+2.5 %). All of that is inside the screening scatter (+-5 %):
  the round found no geometry or option that clearly beats what the sample already had. The full measurement
  decides (queued).
- Fastest alternative level of each parameter, relative to its centre (median over the centres; below 1 = faster):
  accumulator 12.0 (fp32 everywhere), subgroup size 1.49, tile K 1.20, grid X 1.19, grid Y 1.17, tile M 1.14,
  tile N 1.14, `FRAG_LAYOUT` 1.06, `B_COLMAJOR` 1.02, texel-wise staging 1.01, drain mode 1.01, `IMG_W` 1.00,
  `IMG_A` 1.00, `SH_F16V4` 0.96 (4 centres only). So around the best configurations every geometry parameter is at
  a local optimum (the nearest alternative costs 14 to 49 %), and the options are flat within the scatter.

## Part 1: parameter space (static count, `tools/enum_space.py`)

| family | combinations | device | flags | geometry | shared memory | shape | survivors | tile geometries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4w | 75,497,472 | 64,487,424 | 10,515,456 | 292,692 | 76,188 | 0 | 125,712 | 590 |
| 8da4w zpg | 165,888 | 133,632 | 7,632 | 22,002 | 384 | 0 | 2,238 | 594 |
| SDPA QK^T | 9,216 | 3,840 | 0 | 2,832 | 820 | 0 | 1,724 | 628 |
| SDPA attn*V | 4,608 | 1,920 | 0 | 2,023 | 28 | 167 | 470 | 446 |

Columns "device" to "shape" count the combinations removed by each static rule, in that order.
The rules replay without a rejection on the 64 existing variants this device runs (`enum_space.py --check`).

Plan (`tools/plan_space.py`; `results/780m/space/sweep-manifest.csv`): 8da4w (2,238), QK^T (1,724) and attn*V
(470) in full, batches b00 to b07. The 4w rows of that manifest (the staged plan, batches b07 to b10) are not run.

Timing sample, 30 configurations drawn at random (`results/780m/space/sample30-results.csv`, 20:21 to 20:33 UTC):

| family | sample | s per configuration | planned | projected |
|---|---:|---:|---:|---:|
| SDPA (one QK^T and one attn*V token per run: 8 correctness cases, then timing) | 10 runs, 20 tokens | 12.5 per run | 1,724 runs | 6.0 h |
| 8da4w | 10 | 15.8 | 2,238 | 9.8 h |
| 4w, staged (dropped, see the owner decision) | 10 | 18.4 (15.3 to 26.4) | 1,642 | 8.4 h |
| total without 4w | | | | 15.8 h, plus cooling waits and the 06:40 to 07:40 pause |
| 4w, all 125,712 survivors | | 18.4 | 125,712 | **642 h (26.8 days): over the 48 h limit, not started** |

The first sample is in `<artifacts>/superseded/sample30-kernel-name-not-recognised/`: the linear family names
lacked `linear_`, which the microbench needs to report the kernel time.

Found by the sample: `CSH_BAND` 4w tiles whose `SG_GRID_Y * 16 * WG_TILE_N * 2` reaches 64 KiB are refused by
the selector's shared-memory check (`impl/sarc/Select.cpp`, release zone; it does not know the band drain), so
they keep the table kernel and are recorded with `dispatched=0`.

Linear configurations are timed with the microbench's correctness gate off (the gate only knows the table
kernels' small shapes); correctness is checked in the repeat stage. SDPA configurations are checked in every run.

## Part 2: softmax (measured in the microbench; `raw/softmax/`, build `hook2`)

`sarc_sdpa_attn_weights_softmax` op time per layer at S = 2048, profile `780m-refine3` for QK^T and attn*V, two
runs each; all variants reached through `hooks/softmax-name-hook.patch` applied in a scratch tree:

| variant | what | 1B (32 heads) | 3B (24 heads) | 8B (32 heads) | extended correctness tier |
|---|---|---:|---:|---:|---|
| release | three loads of the row prefix, exp twice | 7.19 / 7.21 ms | 5.42 / 5.41 ms | 7.21 / 7.22 ms | 8 of 8 |
| `r1` | one load kept in registers, exp once | 5.94 / 5.95 ms | 4.46 / 4.46 ms | 5.94 / 5.93 ms | 8 of 8 |
| `r2` (dropped) | r1 + in-wave tree reductions, no shared memory | 5.95 ms | 4.45 ms | 5.93 ms | 8 of 8 |
| `r3` | r1 + zero fill bounded to the K-chunks the SARC attn*V kernels read | 4.41 / 4.43 ms | 3.32 / 3.30 ms | 4.45 / 4.42 ms | 8 of 8 |
| `m1` (measurement only, wrong results) | r3 without exp | 4.42 / 4.42 ms | 3.32 / 3.31 ms | 4.42 / 4.45 ms | 0 of 8, as intended |

- r3 is -38.6 % on the kernel (r1 -17.6 %). Removing exp entirely changes nothing (m1 = r3), and neither do the
  barriers (r2 = r1): the kernel is bound by memory traffic, about 284 MB per layer at about 64 GB/s with r3.
- r3 depends on its neighbour: the release softmax comment says attn*V stages every chunk below `context_len`,
  but the SARC attn*V kernel already stops at the chunk holding column `a + M - 1 + input_pos` of its row tile
  (`useful_chunks`), so zeros past that chunk are never read. r3 keeps the full fill unless S and `input_pos` are
  multiples of 256 and head_dim a multiple of 64. With an attn*V that reads whole rows (upstream kernel,
  `ET_VK_DISABLE_COOPMAT`) r3 would be wrong; the microbench pairing check now fails such a pairing.
- Expected end to end (16 / 28 / 32 layers): about -44 ms (1B), -59 ms (3B), -89 ms (8B).

## Next

1. 8da4w screen and confirmation -> the 8da4w kernel per shape; candidate 11 = candidate 10 + those kernels.
   If it is under 2 % too, the stop rule is met.
2. (withdrawn: RGP captures. Owner decision 2026-10-05, see "Host hang and reboot".)
3. Production-diff passes for the confirmed configurations; QK^T / attn*V enumeration and repeat stage.
4. A direct session of the final configuration against `dev/1.5` and against `780m-refine3`.

## Blocking

Nothing.
