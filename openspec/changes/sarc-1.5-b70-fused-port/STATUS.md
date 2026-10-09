# sarc-1.5-b70-fused-port: status

**2026-10-09 07:15 UTC (host clock) — chain 1 is done: hook condition met, SPIR-V identical, baseline within
0.22 % of `s12-final5`, A/A -0.06 % geomean, and the one kernel screen keeps the B580's pair for `b70-fused1`. The
gate of candidate 1 (`tools/chain2.sh`, session `s2-c1`, 7 repeats) is running. No candidate number yet.**

Branch `topic/b70-fused-port` (from `origin/topic/xe2-prefill-refine` at `5617714b0`). Parent of every comparison:
`5617714b0` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5`. Host `fedora-gpu-eval`, card `b70-0`
only (guest PCI `0000:01:00.0`, `ETVK_DEVICE_INDEX=0`, lock `868023e2-0000-0000-0100-000000000000`); the second card
is not used. Artifacts `/home/doremy/hmz-sarc-b70-fused/.artifacts/`.

As found: ANV Mesa 26.2.3 (the first campaign's version), kernel 7.2.9-200.fc44 (host up since 2026-10-08 17:06
UTC; the first campaign's journal did not record its kernel version in `STATUS.md`, so a kernel change since its
`s12-final5` cannot be excluded: the baseline comparison below is the check). GT frequency policy `min_freq` 1200,
`max_freq` 2800, `rp0` 2800, `rpe` 400, `rpn` 400 MHz, `power_profile` `[base] power_saving`; nothing changed.
Device limits read with `vulkaninfo`: `maxComputeSharedMemorySize` 49152, subgroup size 16 to 32 (default 32),
`computeFullSubgroups` true, `maxComputeWorkgroupSubgroups` 64: the same as the B580's record. RAM 46 GB; the six
model files were not in the page cache after the reboot, so decision D5 is applied (each model read before its
cell, both arms alike, resident share recorded per run in `runs.csv`, column `cached_pct`).

## Running now

Detached (`nohup setsid`), one unit at a time: `tools/chain2.sh topic1 s2-c1 b70-fused1 c1-ref`, started 07:10
UTC, status in `.artifacts/logs/chain2-s2-c1.status`: stage `s2-c1` (build `parent` with the parent environment
against build `topic1` with `b70-fused1`), `gate_sdpa.sh` (12 passes x tiers `all` / `extended` / `full` with the
hashes of the test binary and the runner library and every pass's status, unmodified `verify.sh` compared line by
line with `s0-parent-verify`, the timed session with 7 repeats, warm traces, `gate_check.py`), 12 passes each of
tiers `peaked` and `fused` (reported), reference error (`sdpa_ref.sh`), logits probe, `decide.py`, attention
table of the traces, decode comparison, collection. Ends `CHAIN2_DONE s2-c1`; expected about two hours. If the
host reboots: move a half-run `stage/s2-c1` to `superseded/` and start the same command again.

## Chain 1 (05:46 to 07:09 UTC): results

Builds, each from an export of one commit with its submodules (31 trees), shipped SPIR-V golden PASS (53 variants):
`parent` = `5617714b0`, `topic1` = `8ba3607be` (hooks, test blocks, kernel files, `b70-fused` profiles).
`pristine` is a copy of the first campaign's build `parent` (`6a7cc8cc6`; `llama_main` sha256 `45e3242e3881...`,
the binary its `s12-final5` timed), not rebuilt.

**SPIR-V identity** (`tools/spv_identity.sh`, `results/b70/identity/`): `SPV_IDENTITY_OK`. All 1525 shaders of the
parent build are byte-identical in `topic1` (which has 1572); all 47 `sarc_dev_b580_sdpa_fused*` / `..._kvt*`
shaders of `topic1` are byte-identical to the B580 campaign's build `topic6` (`247d08851`, the build its
candidate 1 was gated on).

**Hook condition D4 (nothing selected, nothing changed): met.**
1. `test_sarc_select` built from the parent export and from the `topic1` export: release tables identical output
   (1240 checks, 31 rows, both); with the dev zone and `ET_VK_SARC_UNVERIFIED=1` 1559 checks / 35 rows (parent) and
   1561 / 37 (topic: the two `b70-*` base rows), 215 candidates both. Executables and outputs kept in
   `.artifacts/raw/d4/` (outputs and hashes in `results/b70/d4/`).
2. `spirv_golden.py` PASS, 53 shipped variants, on `parent` and on `topic1`.
3. Unmodified `verify.sh` with no environment on `topic1` (`s0-topic1-noenv`) against the parent's
   (`s0-parent-noenv`): `VERIFY_SAME`, 34 lines, rates removed, kernel names included
   (`stage/s0-topic1-noenv/verify_diff.txt`).

**Parent snapshot `s0-parent-verify`** (parent environment `xe2-refine5`): `CONTROL_RECORDED`, one device-status
item (`correctness rc=1` with 28 of 28 numeric and 4 of 4 rank-3 cases PASSED); SDPA tiers 4 / 8 / 4 cases with 0
mismatches on the parent's three kernels. The two no-environment snapshots show the five status items the first
campaign recorded for pristine `dev/1.5` on this card.

**Baseline and A/A, session `s1-aa`** (06:44 to 06:56 UTC): build `parent` against build `topic1`, both with the
parent environment; median of the first 5 valid runs per arm, arms interleaved; 60 timed runs, none rejected;
tok/s (recomputed from `runs.csv`):

| cell | parent | topic, same environment | A/A | expected (`s12-final5`) | parent vs expected | spread parent / topic |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 17964.90 | 17964.90 | 0.00 % | 17964.90 | 0.00 % | 0.87 / 1.75 % |
| 1B 8da4w | 20686.90 | 20686.90 | 0.00 % | 20686.90 | 0.00 % | 7.48 / 1.00 % |
| 3B 4w | 7529.41 | 7529.41 | 0.00 % | 7529.41 | 0.00 % | 0.00 / 0.37 % |
| 3B 8da4w | 9570.09 | 9570.09 | 0.00 % | 9570.09 | 0.00 % | 2.75 / 2.30 % |
| 8B 4w | 3385.12 | 3379.54 | -0.16 % | 3379.54 | +0.17 % | 1.49 / 0.82 % |
| 8B 8da4w | 4491.23 | 4481.40 | -0.22 % | 4481.40 | +0.22 % | 0.87 / 0.44 % |

A/A geomean -0.06 %; baseline within 0.22 % (limit 3 %); next token SAME in all six cells on the three prompts.
Equal medians are equal millisecond counts: the runner's timer has a 1 ms step (114 ms for 1B 4w, 0.9 % a step).
Per run: foreign engine time 0.00 % in all 60, at least one guard poll while the runner executed, 9 to 45 clock
samples in the prefill window, model file 100 % resident before every run, throttle reasons in the samples `none`
and `pl2` only. Calibration (`tools/thresholds.txt`, dated block, committed `4172bc183` before candidate 1 was
timed): `CLKMIN` 2457 MHz, idle 58 C, **7 repeats** (the parent arm of 1B 8da4w spread 7.48 %: one run of five
read 19140 tok/s, 107 ms against 99 ms).

**The one kernel screen, `screen1-select`** (06:56 to 07:09 UTC, build `topic1`, 3 rounds, cooled before every
run; kernel time per layer at S = 2048 in us, fused kernel + copy pass, for the parent QK^T + softmax + attn*V;
each round's value; `results/b70/screens/screen1-select{,-runs}.csv`):

| head_dim | profile | 1B | 3B | 8B | vs the parent's three kernels |
|---|---|---|---|---|---|
| | parent `xe2-refine5` | 1485 / 1481 / 1481 | 1284 / 1286 / 1285 | 1685 / 1697 / 1691 | 1.00x |
| 64 | `d64_t32x32s32m8ro` (the 780M's) | 4255 / 4281 / 4142 | | | 0.35x |
| 64 | **`d64_t16x64s16m8g4roj`** (incumbent, the B580's) | **486 / 483 / 487** | | | **3.05x** |
| 64 | `d64_t16x64s16m8g4oj` | 553 / 551 / 550 | | | 2.69x |
| 128 | `d128_t16x64s32m8ro` (the 780M's) | | 9569 / 9606 / 9528 | 12577 / 12565 / 12604 | 0.13x |
| 128 | **`d128_t16x128s16m8g8oj`** (incumbent, the B580's) | | **750 / 749 / 748** | **969 / 966 / 967** | **1.72x / 1.75x** |
| 128 | `d128_t16x64s16m8g4oj` | | 824 / 813 / 819 | 1047 / 1047 / 1043 | 1.57x / 1.61x |

No screened variant is faster than the incumbent of its head_dim in any round (the nearest is 8 to 14 % slower),
so by the `kernel_screen` rule **`b70-fused1` is the B580's pair**, as committed. Same ranking as the B580's
screen 5 in every row.

## What was brought from the B580, pinned

Source: `/mnt/linux-share/hmz-campaigns/b580-fused/executorch`, branch `topic/b580-fused-port`, **pinned at
`cea76c634`** (fetched as `refs/remotes/b580/topic/b580-fused-port`; its head is not followed).

| commit here | what | source |
|---|---|---|
| `cfa31c1d2` | release-zone hook D4.1, softmax variant name | cherry-pick of `fab9606c3` (B580 branch; from `b969e8f1c2`) |
| `cbbe36e0c` | release-zone hook D4.3, entry point of the fused attention node | cherry-pick of `0ffc84a2d` (B580 branch; from `1c8861aa7e`) |
| `087d4c4a9` | `test_llama_microbench.cpp`: insert-only `4070ti-fused` blocks (98 added lines, 0 removed) | `origin/topic/4070ti-fused-port`, `ed8b5af91`, unchanged |
| `7d7877980` | `sarc_dev_b580_sdpa_fused.{glsl,yaml}`, `sarc_dev_b580_sdpa_kvt.{glsl,yaml}`, `impl/sarc_dev/b580/SdpaB580Fused.cpp` | `cea76c634`, files unchanged |
| `dfbaecca1` | `impl/sarc_dev/B70Sdpa.cpp` (base rows for `b70-*` profile names) and the `b70-fused` blocks of `Overrides.cpp` | new here |

Notes on these choices:

- The fused hook was cherry-picked alone first and conflicted in `Select.h` (its context is the `softmax_variant`
  field), so both hooks are carried exactly as the B580 branch carries them; the release zone of this branch then
  differs from the parent by the same 57 added lines as the B580's. Nothing here sets `softmax_variant`.
- The test blocks are the 4070 Ti port's, not the B580's: the B580 changed lines of the shared test file in place
  (task section 4.3 forbids that here). Consequence: the `fused` tier has five cases here (the 4070 Ti's adds
  `fused_s32`), four on the B580; that tier is reported, not a gate item.
- The selector is the B580's file, reused unchanged: it is device-neutral for Intel (it asks only for active SDPA
  rows and for the variant list of the profile; no device string). The two functions it calls
  (`sdpa_fused_variants_b580`, `register_sdpa_fused_b580`) are defined in the `b70-fused` block here. A merge with
  the B580 branch will meet two definitions of them and must join the two profile tables; that is deliberate
  (two silent registrations on one hook would be worse).
- The fused yaml has no generator (`gen_b580.py` does not mention it); nothing was generated here.
- **The task file names the pair `d64_t16x32s16m8ro` / `d128_t16x64s16m8ro`; that was the B580's first smoke
  definition. At the pinned commit `b580-fused1` is `d64_t16x64s16m8g4roj` + `d128_t16x128s16m8g8oj`** (its
  `STATUS.md`, "selected by screen 5", and `kFusedB580Pair` in its `Overrides.cpp`). `b70-fused1` is that pair.

## Shared-memory reading (R7), done before any gate

Read by me on `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.glsl` as built for the two variants of `b70-fused1`
(`MULTI_SG`, `ONLINE`, `QK_J_OUTER`; head_dim 64: 16 rows, 64-column blocks, 4 subgroups, `SEGS` 4, `P_STRIDE` 9;
head_dim 128: 16 rows, 128-column blocks, 8 subgroups, `SEGS` 8, `P_STRIDE` 17). Lane = `gl_SubgroupID * 16 +
gl_SubgroupInvocationID` owns row `lane % 16` and segment `lane / 16`: one lane per (row, segment). Every barrier
is `memoryBarrierShared(); barrier()`.

| slot | writers | what orders the reads |
|---|---|---|
| `Psh` scores, tile (i, j) | one `coopMatStore` by the subgroup with `j % G == gl_SubgroupID` (one column each in both variants); tiles are disjoint | `SYNC()` after `qk_block`, then each lane reads its own segment |
| `Rsh[row * SEGS + seg]` | the one lane of (row, seg) | `SYNC()` before the row's lanes read all segments, `SYNC()` after |
| `Gsh` | `atomicOr` by any lane; plain clear by lane 0 only | `SYNC()` between the `atomicOr`s and the one read per lane; the clear follows the `SYNC()` after the divisor stores, which every lane enters after its read; a `SYNC()` follows the clear |
| `Dsh[row * 4 + j]` | the lane of the row whose segment is j (segments 0 to 3; with 8 segments the upper four write nothing) | `SYNC()` before the `coopMatLoad`s, `SYNC()` after |
| `Psh` e values | each lane its own segment | `SYNC()` before `av_block` loads the block, `SYNC()` after it and before the next block's score stores |
| `Psh[row * P_STRIDE + j]` final sums | as `Dsh` | `SYNC()` before the `coopMatLoad` of `den`; the last loads of `Psh` are a `SYNC()` earlier |
| `t_output` | one `coopMatStore` per tile by the subgroup that owns that head_dim slice; other workgroups own other rows or heads | not read |

No slot has two plain writers in a phase. Control flow around every `barrier()` is uniform for the workgroup: the
two early returns precede the first barrier and depend on `gl_WorkGroupID`, uniforms, `gl_NumSubgroups` and
`gl_SubgroupSize`; the block count is per workgroup; the rescale branch tests `Gsh`, read by every lane after the
same barrier. The values written by the lanes of one row to `Dsh` are computed from maxima already reduced over
the row, so the slot's single writer holds the row's value. The copy pass `sarc_dev_b580_sdpa_kvt` has no shared
memory. This agrees with the B580's table; its reading applies here because the limits it rests on are the same
(49152 bytes of shared memory against 3588 / 5892 used; required subgroup size 16 in both yaml entries; subgroup
range 16 to 32).

Specification sentences relied on, each found again on this host through the `vulkan-docs` server (exact-phrase
`search_docs`, 05:47 UTC):

- "... all compute shader invocations for a single workgroup must enter it before any will continue beyond it."
  (`glsl/latest/builtinfunctions.md`, https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html)
- "Full subgroups are required when the X dimension of the workgroup size is a multiple of the reported value of
  the SubgroupSize built-in, and one of the following is true: ..." (`spec/latest/shaders.md`,
  https://docs.vulkan.org/spec/latest/chapters/shaders.html). As the B580 campaign found, none of the listed
  conditions holds for these pipelines (SPIR-V 1.3, no full-subgroups flag), so the kernel's run-time check
  (`gl_NumSubgroups == G`, `gl_SubgroupSize == 16`, NaN rows otherwise) is what the lane bijection rests on. That
  campaign's finding F1 (the cooperative-matrix pipelines of the branch, shipped ones included, lack the flag the
  specification requires) applies here unchanged and is not repeated as a question.
- "... a single implementation-dependent invocation within the instance of the matrix's scope performs a
  non-atomic store to that memory location." (`spec/latest/memorymodel.md`,
  https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)
- "If the shader was created with a required subgroup size, the SubgroupSize decorated variable will match that
  value." (`refpages/latest/SubgroupSize.md`)

The remaining sentences of the B580's list (atomics, the data-race definition, uniform control flow) were read in
its `STATUS.md` at `cea76c634` and not searched again here.

## Next

Read chain 2 (gate, reference error, probe, decision). Then read the B580 campaign's `STATUS.md` for its
candidate 2 (task section 6.4), then the closing on the build of the committed head.

## Decision needed from the owner

Nothing.

## Blocking

Nothing.
