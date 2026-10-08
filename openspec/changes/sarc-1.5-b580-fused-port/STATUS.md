# sarc-1.5-b580-fused-port: status

**2026-10-08 21:55 UTC — the straight port is correct but not fast on this card: with the 780M's tile shapes the
fused kernel is 7 times slower than the parent's three kernels for head_dim 128 and 1.25 times slower for
head_dim 64 (screen 1, round 1). Its time follows the number of matrix tiles it keeps live. An 8-row head_dim 64
variant is 1.5 times faster than the three kernels; no head_dim 128 variant of the first set is. Register-saving
variants are being built. No candidate timed end to end yet.**

Branch `topic/b580-fused-port`, parent `51d9d757f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`.
Host `fedora` (the owner's desktop), Arc B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`. Artifacts `/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.
As found: ANV Mesa 26.2.3 (the first campaign's version), kernel 7.2.8-200.fc44; GT frequency policy
`min_freq` 1200, `max_freq` 2850, `rp0` 2850 MHz, `power_profile` `[base] power_saving`, nothing changed; the
desktop session idle and locked at 20:00 UTC.

## Running now

Detached `tools/chain4.sh` (since 21:52 UTC; `.artifacts/logs/chain4.status`, ends `CHAIN4_DONE` or
`CHAIN4_STOPPED`): build `topic3` (= `0ac2250f5`, container build); one correctness pass of tiers `all`,
`extended`, `peaked`, `fused` for each of the 14 new variants (GPU, not timed, `raw/c1-smoke2/`); one-round
look `screen2-fused` (GPU, timed, each run behind the idle wait). Chains 1 to 3 have ended (chain 3 stopped by
the actor after round 1 of screen 1, see below).

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

## Shared-memory reading of the fused kernel (R7; written before any gate)

A workgroup is one subgroup of `SUBGROUP_SIZE` lanes. Lane `id` owns row `id % WG_TILE_M` and segment
`id / WG_TILE_M` of every block (a bijection between lanes and (row, segment) pairs).

| slot | writer | cross-lane reader | what orders the read after the write |
|---|---|---|---|
| `Psh` scores, tile (i, j) | `coopMatStore` in `qk_block` (one subgroup-wide store per tile, tiles disjoint) | each lane reads its own segment of its own row | `memoryBarrierShared(); subgroupBarrier();` right after `qk_block` |
| `Rsh[row * SEGS + seg]` (block maximum, later the row sum) | the one lane that owns (row, seg) | the lanes of the same row read all `SEGS` slots | barrier pair between the store and the reads, and another after the reads before the slot is written again |
| `Dsh[row * 4 + j]` (rescale divisors, one-pass form) | the lane of the row with `j % SEGS == seg`: one writer per slot | `coopMatLoad` of the divisor tiles | barrier pair after the stores, barrier pair after the loads |
| `Psh` e values, own segment | the one lane that owns (row, seg) | `coopMatLoad` in `av_block` | barrier pair before `av_block`, barrier pair after it (before the next block's `coopMatStore`) |
| `Psh[row * P_STRIDE + j]`, final divisors | the lane of the row with `j % SEGS == seg` | `coopMatLoad` of `den` | barrier pair after the stores; the last reads of `Psh` (`av_block`) are a barrier pair earlier |

No slot has two writers in one phase, so no elected lane is needed. Every barrier sits in control flow that is
uniform for the subgroup (`num_blocks` and `s_base` are per workgroup; the rescale branch is taken on
`subgroupAny`). The one-subgroup assumption: the pipeline is created with the variant's required subgroup size
(yaml `SUBGROUP_SIZE`, as for the Xe2 kernels) and the node launches a local size equal to it; the release-zone
pipeline code does not set the full-subgroups flag, so the kernel itself checks `gl_NumSubgroups == 1` and
`gl_SubgroupSize == SUBGROUP_SIZE` and writes NaN rows otherwise (one writer per row), which every correctness
tier would report.

## Next

After chain 2: fix the variants of `b580-fused1` by the screen rule of `thresholds.txt` (a rebuild under a new
tag if they change), then the gate of candidate 1 (`gate_sdpa.sh`, reference-error run, logits probe, decision).

## Blocking

Nothing.
