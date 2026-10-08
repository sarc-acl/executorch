# sarc-1.5-b580-fused-port: status

**2026-10-08 20:00 UTC — candidate 1 (`b580-fused1`, one-pass fused kernel) is correct at kernel level in a
first smoke pass of all five tiers. Not gated, nothing timed yet.**

Branch `topic/b580-fused-port`, parent `51d9d757f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`.
Host `fedora` (the owner's desktop), Arc B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`. Artifacts `/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.
As found: ANV Mesa 26.2.3 (the first campaign's version), kernel 7.2.8-200.fc44; GT frequency policy
`min_freq` 1200, `max_freq` 2850, `rp0` 2850 MHz, `power_profile` `[base] power_saving`, nothing changed; the
desktop session idle and locked at 20:00 UTC.

## Running now

Detached, one unit at a time (status lines in `.artifacts/logs/chain1.status` and `chain2.status`; each chain
ends with `CHAINn_DONE` or `CHAINn_STOPPED`):

- `tools/chain1.sh` (since 19:31 UTC), remaining units: parent snapshots `s0-parent-verify` (parent
  environment) and `s0-parent-noenv`; hook condition `s0-topic1-noenv` against `s0-parent-noenv`; baseline + A/A
  with calibration, session `s1-aa` (GPU, timed: no build on this host while it runs).
- `tools/chain2.sh`, waiting for chain 1: kernel-level screen `screen1-fused` of the 16 fused variants against
  the parent's three attention kernels, 3 rounds, cooled before every run.

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
- Hook condition D4, first part: `test_sarc_select` built from the parent export and from the branch head gives
  the same output for the release tables (1240 checks, 31 rows) and with the dev zone and
  `ET_VK_SARC_UNVERIFIED=1` (1562 checks, 37 rows, 213 candidates): `.artifacts/raw/d4/select-*.txt`. The
  `verify.sh` part (no environment, line by line) is a unit of chain 1.
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
