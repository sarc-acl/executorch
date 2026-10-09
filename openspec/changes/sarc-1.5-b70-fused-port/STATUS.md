# sarc-1.5-b70-fused-port: status

**2026-10-09 06:00 UTC (host clock) — preparation done, first detached chain running (builds, parent snapshots,
baseline + A/A, the one kernel screen). No timed number yet.**

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

Detached (`nohup setsid`), one unit at a time: `tools/chain1.sh 8ba3607be`, status in
`.artifacts/logs/chain1.status`, started 05:46 UTC. Units: builds `parent` (`5617714b0`) and `topic1`
(`8ba3607be`) from exports, logits probes; `pristine` = a copy of the first campaign's build `parent` (the binaries
its `s12-final5` timed); SPIR-V identity; `test_sarc_select` for the hook condition; snapshots `s0-parent-verify`
(parent environment), `s0-parent-noenv`, `s0-topic1-noenv`; session `s1-aa` (parent against `topic1`, both with the
parent environment, `--calibrate`); screen `screen1-select` (3 rounds). Ends `CHAIN1_DONE`. If the host reboots:
read the status file and start the chain again with the same commit; finished builds and snapshots are skipped, a
half-run `stage/s1-aa` goes to `superseded/` first.

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
`search_docs`, 06:00 UTC):

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

Read chain 1: SPIR-V identity, hook condition, baseline against `s12-final5` (3 %), A/A; append the calibration to
`tools/thresholds.txt`; read the screen by the `kernel_screen` rule; then the gate of candidate 1 (`b70-fused1`).

## Decision needed from the owner

Nothing.

## Blocking

Nothing.
