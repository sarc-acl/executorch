# sarc-1.5-b580-fused-port: status

**2026-10-09 01:26 UTC — the desktop was rebooted at 01:17 UTC (an orderly `systemd` reboot, not started by
this campaign; kernel 7.2.8-200 -> 7.2.9-200, Mesa 26.2.3 unchanged, card present, GT frequency policy as
before). It stopped chain 8 inside the logits probe of `s2-c1` and chain 9 while it waited; nothing measured was
lost (all 450 saved logits files have the full size). `tools/chain10.sh` resumed at 01:21 UTC. Candidate 1
(`b580-fused1`, the fused attention kernel in a multi-subgroup form): `GATE_PASS`, a plain pass (next token SAME
in all 18 cell x prompt comparisons), **+9.18 % geomean** over the parent `b580-refine3` (1B +17.1 / +19.0 %, 3B
+5.4 / +6.8 %, 8B +3.5 / +4.3 %; 4w / 8da4w), 84 timed runs, all valid, on the desktop in use. Its error against
the fp32 reference is not larger than the parent's in all 4 production-shape cases and all 8 `extended` cases;
in 1 of 5 synthetic `peaked` cases its maximum error is larger (rms smaller). The specification quotes are in;
no quote contradicts the kernel. Candidate 2 follows in the same chain.**

Branch `topic/b580-fused-port`, parent `51d9d757f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`.
Host `fedora` (the owner's desktop), Arc B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`. Artifacts `/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.
As found: ANV Mesa 26.2.3 (the first campaign's version), kernel 7.2.8-200.fc44; GT frequency policy
`min_freq` 1200, `max_freq` 2850, `rp0` 2850 MHz, `power_profile` `[base] power_saving`, nothing changed; the
desktop session idle and locked at 20:00 UTC.

## Running now

Detached, one GPU job at a time, no idle wait (owner decision 2026-10-09 00:22 UTC):

- `tools/chain10.sh` (started 01:21 UTC; `.artifacts/logs/chain8-s2-c1.status`, then `chain9.status`): the units
  the reboot cut off: the logits probe of `s2-c1` (resumes; saved prompts are skipped), `decide.py`, the decode
  comparison, collection; writes `CHAIN8_DONE s2-c1`; then `tools/chain9.sh` unchanged: candidate 2, session
  `s3-c2` (`b580-fused2` against candidate 1, both build `topic6`): softmax look, `gate_sdpa.sh` with 7 repeats
  (timed), reference error, probe, decision, decode. Ends `CHAIN9_DONE`.

If the machine reboots again, both are gone; restart with `chain10.sh` (every unit is resumable or is repeated
whole; a half-run session directory goes to `superseded/` first).

No build may start on this host while they run.

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

## Result so far: candidate 1, `b580-fused1`: GATE_PASS, +9.18 % geomean

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
`gl_NumSubgroups == G` and `gl_SubgroupSize == 16` and writes NaN rows otherwise (one writer per row), which
every correctness tier would report. The copy pass `sarc_dev_b580_sdpa_kvt` has no shared memory.

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
     specification does not guarantee full subgroups for this kernel**; the run-time check is what the kernel
     rests on. It is sufficient: the local size is G x 16 in X (`SdpaB580Fused.cpp`, `LocalWorkGroup(wg, 1, 1)`),
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

**Finding F1 (not a change of this campaign; reported for the owner).**
* "VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770 Any pipeline containing a shader with
OpTypeCooperativeMatrixKHR or OpCooperativeMatrix*KHR instructions must be created with the
VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be version 1.6 or
greater" (`refpages/latest/RuntimeSpirv.md`). Every cooperative-matrix pipeline of this branch is SPIR-V 1.3
without the flag: the shipped release kernels, the parent's Xe2 attention kernels and the fused kernel alike
(checked on `sarc_sdpa_qk_coopmat_xe2c_...` and both fused variants of `build/topic6`). Setting the flag is a
change of the release-zone pipeline code (`vk_api/`), which decision D4 does not cover, and SPIR-V 1.6 needs an
instance of Vulkan 1.3; neither was done. ANV launches full subgroups for these kernels in every run measured
(the fused kernel would have written NaN rows otherwise, and every correctness tier passes).

## Next

Read the probe, the decision and the decode comparison of candidate 1; read the gate of candidate 2
(`b580-fused2` = candidate 1 + the 4070 Ti campaign's fp32 no-tail softmax for the calls the fused kernel does
not take). Then stop by the task (two candidates at most) and close: build the committed head, full gate of the
final stack, one timed session against the parent of section 3 and one against the pristine parent `6a7cc8cc6`.

## Blocking

Nothing.
