# sarc-1.5-b580-fused-port: status

**2026-10-08 19:31 UTC — port written, not yet built or run on the card. Nothing measured yet.**

Branch `topic/b580-fused-port`, parent `51d9d757f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`.
Host `fedora` (the owner's desktop), Arc B580 = PCI `0000:03:00.0`, Vulkan device 0, `ETVK_DEVICE_INDEX=0`, lock
`86800be2-0000-0000-0300-000000000000`. Artifacts `/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.

## Running now

Detached chain `tools/chain1.sh` (status lines in `.artifacts/logs/chain1.status`, ends with `CHAIN1_DONE` or
`CHAIN1_STOPPED`), one unit at a time:

1. build of the parent `51d9d757f` as `build/parent2` and `build/parent2-traced` (`.artifacts/logs/build-parent2.out`);
2. build of the topic commit as `build/topic1`;
3. smoke run of the fused kernel, one pass of each SDPA correctness tier with `b580-fused1` (`raw/c1-smoke/`; not a gate);
4. parent snapshots `s0-parent-verify` (parent environment) and `s0-parent-noenv`;
5. hook condition D4: `s0-topic1-noenv` against `s0-parent-noenv`, line by line (`tools/verify_diff.py`);
6. baseline + A/A with calibration, session `s1-aa` (GPU, timed: do not build on this host while it runs).

The first actor run of this campaign was stopped at 19:22:50 UTC, two and a half minutes into its build of tag
`parent`; that build died with it at about 70 % and is kept, unused, under
`.artifacts/superseded/parent-build-interrupted-20261008T1922Z/`. Nothing was measured with it.

## Done so far (no measurement)

- Change directory, tools copied and adapted, thresholds fixed (`7112e8930`).
- Release-zone hooks, each its own cherry-picked commit: `fab9606c3` (softmax variant name, D4.1, from
  `b969e8f1c2`) and `0ffc84a2d` (fused attention entry point, D4.3, from `1c8861aa7e`). The D4 evidence that
  nothing changes while nothing is selected is still owed (needs the builds).
- Candidate 1 sources: `glsl/sarc_dev/sarc_dev_b580_sdpa_fused.{glsl,yaml}` (the `fused3sb` kernel for the
  8 x 16 x 16 matrix shape, packed form only, one-pass and two-pass variants), `sarc_dev_b580_sdpa_kvt.{glsl,yaml}`
  (the 780M's copy pass, unchanged), `impl/sarc_dev/b580/SdpaB580Fused.cpp` (the node), a `b580-fused` block in
  `impl/sarc_dev/Overrides.cpp` (profile `b580-fused1` = `b580-refine3` + one fused variant per head_dim; single
  name, no second variable; `b580-fused-<variant>` screening profiles), and the microbench's fused-kernel
  bookkeeping with the 780M's `peaked` and `fused` correctness tiers. Compiles (host `glslc` for the 17 shader
  variants, `g++ -fsyntax-only` for the two C++ files); `sarc/tools/check.sh --no-build`: PASS.

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

Parent snapshots (`s0-parent-verify`, `s0-parent-noenv`), build of the topic head, D4 evidence, baseline + A/A
(`s1-aa`) with calibration, then kernel-level correctness and the variant screen of candidate 1.

## Blocking

Nothing.
