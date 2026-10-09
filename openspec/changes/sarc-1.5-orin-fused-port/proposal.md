# sarc-1.5-orin-fused-port

Second campaign on the Jetson Orin Nano (`duck-naughty`). Branch `topic/orin-fused-port`, forked from
`topic/orin-prefill-refine` at `8973ced76`. It ports one thing on top of the first campaign's final stack: the
fused attention kernel of the Radeon 780M (third structure, with the RX 7600's subgroup barriers). A port, by
owner decision N1: no sampled search, no enumeration, no tile sweep; two candidates at most.

Parent of every comparison: `8973ced76` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack, `sarc-1.5-orin-prefill-refine/proposal.md`, "Outcome"). The pristine state is
the same commit with no environment.

Day-to-day state is in `STATUS.md`. This file is completed as the campaign proceeds.

## Thresholds (fixed before the first measurement)

`tools/thresholds.txt`, committed before any run of this campaign:

- baseline: every cell of the parent within 3 % of `s10-final` (1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
  8B 295.53 / 269.19 tok/s, 4w / 8da4w);
- noise band +-2 %; median of 5 valid interleaved runs per arm and cell;
- clock floor: floor(0.97 x the lowest per-cell median clock of this campaign's own A/A session), device-wide;
- a run needs at least 5 clock samples in its prefill window (0.1 s sampler; the shortest prefill is 1.2 s);
- kernel screen margin 3 % in every round;
- reference-error rule (D3): "the parent" is this campaign's parent, the tuned stack's arithmetic; the stock
  kernels' error is reported beside it, not judged;
- candidate 2 only if the unpacked one-pass form is at least 3 % faster at kernel level in every round.

## Tools: what differs from the sibling (`sarc-1.5-orin-prefill-refine/tools`)

Copied, originals untouched. Not carried over: the `chain*.sh` of the first campaign, its generators
(`gen_orin_*.py`, `orin_refine.py`), the linear screen, phase-timing, decode and second-device tools and the old
local hook patch. Byte-identical to the sibling's and not listed: 37 files (`gl.sh`, `stage.sh`, `gate.sh`,
`timed.sh`, `session.sh`, `noenv_verify.sh`, `probe_*.py`, `probe_run.sh`, `ref_error_rule.py`, `sdpa_err.sh`,
`sdpa_screen.sh`, `shipped.py`, `summarize.py`, `nexttoken.py`, `calibrate_clock.py`, `roof.sh`, `roof_fast.py`,
`roof_util.py`, `mktree*.sh`, `build-extra.sh`, `wsrun.sh`, ...).

| tool | change |
|---|---|
| `common.sh` | device directory `~/hmz-sarc-orin-fused`, change directory `sarc-1.5-orin-fused-port`, workstation root `/mnt/linux-share/hmz-campaigns/jetson-fused` with the artifact directory `.artifacts` itself, `PARENT_COMMIT=8973ced76`, campaign tag `orin-fused-port`; the second Orin is removed (not available); `hold_wait` (coordinator hold, D6) called by `take_lock`; `throttle_read` and a sixth field in every clock sample: the state of the module's 12 thermal cooling devices (the fan left out) |
| `runrow.py`, `e2e5.sh` | a timed run needs at least 5 clock samples (was 2; `thresholds.txt`), and is invalid with a nonzero throttle state in its window (`thermal_throttle`) or without the record (`no_throttle_record`); the model file is read into the page cache before each cell (D5) and the load time is recorded; an 8B run is not started with less than 5000 MB available; the session log lists the cooling devices and trip points |
| `gate_sdpa.sh`, `gate_check.py`, `test_gate_check.py` | the SDPA step runs 12 passes of the tier `all` as well as `extended` and `full`, and under an `orin-fused` profile 3 passes of the 780M's tiers `peaked` and `fused`; a case the fused node serves must name that fused kernel and no QK^T / softmax / attention x V kernel, every other case the three kernels and the profile's softmax; the fused-attention banner must match the environment; 54 unit tests |
| `gatelib.sh`, `parent_verify.sh` | `PARENT_CTL_NAME` selects the control (`s0-parent-verify`: the tuned parent; `s0n-noenv`: nothing selected); the parent control takes an environment, because this campaign's parent is a tuned stack; `hold_wait` before `verify.sh` |
| `trace_analyze.sh`, `trace_kernels.py` (new) | the ETDump analysis uses the first campaign's devtools venv read-only; `trace_kernels.py` adds `attention.csv` and `kernels.csv` (the fused kernel and its copy pass by name) |
| `sdpa_screen_summary.py`, `collect.sh` | under a fused profile the per-layer total is the copy pass + the fused kernel, every round listed; the pre-checks, the probes and the error files of this campaign are collected |
| `pull.sh`, `build-orin.sh`, `deploy.sh`, `drun.sh`, `trace.sh` | paths and comments; the second Orin's mirror removed; D5 in `trace.sh` |
| `jetson-cross/{container,build}.sh`, `build-orin.sh` | option `GLSLC_DIR` (off by default): compile the shaders with a `glslc` given by directory instead of the image's; used for the unmeasured builds `parentg` / `topic4p` only ("Known limitation") |
| new | `thresholds.txt`; `HOLD.md`; `buildq.sh`, `extraq.sh` (build queue under the desktop build lock); `memprobe.sh` (memory of the K / V copies); `chain1.sh` to `chain10.sh` (the detached device queues as they ran; `chain10` was stopped by the owner in its first step, "Known limitation"; `chain5`, `chain6`, `chain7` were ended before they started a job) |

## Release-zone hook (owner decision 2026-10-05)

Commit `9d91480b2` is the 780M campaign's `1c8861aa7e` cherry-picked unchanged (same patch-id; hook D4.3, the
entry point of the fused attention node): 49 added lines, none removed, in `impl/SDPA.cpp` (7),
`impl/sarc/SdpaCoopmat.cpp` (21), `impl/sarc/SdpaCoopmat.h` (8), `impl/sarc/Select.h` (13); the diff is
`git show 9d91480b2`. `Override::sdpa_fused_add` and `Override::sdpa_fused_serves` are null by default;
`sarc::add_sdpa_fused` is called once after the three SDPA nodes of an LLM-mode op and does nothing without the
first; `sarc::sdpa_fused_skip` gives the three nodes an empty launch geometry only for a call the second says the
fused node serves. **It is larger than a switch and stays subject to the owner's review before any promotion.**
On this branch the dev zone sets the two fields only when a fused variant is named (profile `orin-fused1` or
`orin-fused2`, or the kernel-timing variable `ET_VK_SARC_ORIN_SDPA_FUSED`), so every other configuration of a dev
build has them null as well. No other release-zone file is changed by this campaign
(`git diff --name-status 8973ced76 HEAD`: the four files above; everything else is dev zone or this change
directory).

## The port

What runs: for an LLM-mode attention call with fp16 buffers whose shape fits (S a multiple of the row tile and of
the column tile, `input_pos` a multiple of the column tile, head_dim 64 or 128), the node
`impl/sarc_dev/orin/SdpaOrinFused.cpp` dispatches `sarc_dev_orin_sdpa_kvt` (tile-packed copies of K and V for the
visible context) and then `sarc_dev_orin_sdpa_fused3sb_<variant>`, and the QK^T, softmax and attention x V nodes
dispatch nothing. Every other call (decode, an unaligned prompt) runs the parent's three kernels with the softmax
`orin_g64`, exactly as the parent does: an `orin-fused*` profile carries the preferences of `orin-refine5` and
names that softmax itself, so one variable selects the whole stack.

| head_dim | models | rows x columns per workgroup | `orin-fused1` (candidate 1) | `orin-fused2` (candidate 2) |
|---|---|---|---|---|
| 64 | 1B | 32 x 32, 32 invocations | `fused3sb_d64_t32x32g11s32rko` (one pass) | `fused3sb_d64_t32x32g11s32rk` (two passes) |
| 128 | 3B, 8B | 16 x 64, 32 invocations | `fused3sb_d128_t16x64g11s32rko` (one pass) | the same |

`r`: Q tiles kept in registers; `k`: tile-packed K and V; `o`: one pass with a running row maximum. The yaml also
builds the unpacked forms (`ro`, `r`), reachable only through `ET_VK_SARC_ORIN_SDPA_FUSED` for kernel timing.

What changed against the source kernels:

| item | source | here | why |
|---|---|---|---|
| fused kernel from `#version` on | `sarc_dev_780m_sdpa_fused3sb.glsl` of the RX 7600 branch (`b3bb758e38`): the 780M's third structure with `subgroupBarrier()` after every `memoryBarrierShared()` | the same text plus five lines after the row-range check: `if (gl_NumSubgroups != 1u \|\| gl_SubgroupSize != SUBGROUP_SIZE) return;` (commit `0bed38090`, as the 4070 Ti port has it) | the shared memory is private to one subgroup only if the workgroup is one full subgroup. The pipeline requires subgroup size 32 but not full subgroups (release zone), so the kernel checks it and writes nothing otherwise: the correctness tiers then fail instead of the kernel racing silently |
| constants (`MMA`, tiles, `SUBGROUP_SIZE`, `P_STRIDE`, spec constants) | | unchanged | fp16 16 x 16 x 16 matrices with fp32 result and subgroup 32 exist on this device (`results/orin/vk-caps-first-campaign.txt`) |
| copy passes | `sarc_dev_780m_sdpa_{kvt,vt}.glsl` (780M) | the same text from `#version` on; workgroup 8 x 8 | |
| names | `sarc_dev_780m_sdpa_` | `sarc_dev_orin_sdpa_` | separable content; `Sdpa780mFused.cpp` is not edited |
| selector | `impl/sarc_dev/780m/Sdpa780mFused.cpp` (`8518659efb`) | `impl/sarc_dev/orin/SdpaOrinFused.cpp`: node, launch geometry and fit rule unchanged; variants named by profile | one name per configuration |
| `impl/sarc_dev/Overrides.cpp` | | two appended blocks, `orin-fused profiles` and `orin-fused override` (16 lines) | R3 |
| `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` | | seven insert-only blocks delimited by `// >>> orin-fused <id>` (`ca62778e6`); no line of the parent's file removed or modified | R3. The first form (`0f14f2a1a`, builds `topic1`, `topic2`) changed 9 existing lines; found by reading the 4070 Ti port's review |

Shared memory of one workgroup:

| array | elements | d64 `t32x32` | d128 `t16x64` | holds |
|---|---|---:|---:|---|
| `Psh` | `WG_TILE_M x max(WG_TILE_N / 8 + 1, 4)` uvec4 | 32 x 5 x 16 = 2560 B | 16 x 9 x 16 = 2304 B | the block's scores (fp16), then e (fp16); at the end the row sums as 16 fp32 per row |
| `Rsh` | `WG_TILE_M x SEGS` float | 128 B | 128 B | per-invocation block or row maximum, at the end the per-invocation row sum |
| `Dsh` (one-pass forms only) | `WG_TILE_M x 4` vec4 | 2048 B | 1024 B | per-row rescale divisor exp(new maximum - old maximum), 16 fp32 per row |
| total, one pass / two passes | | 4736 / 2688 B | 3456 / 2432 B | |

The reading of the kernel for unsynchronised shared writes (who writes each slot, what orders each read by
another invocation), the specification sentences it rests on and the SPIR-V barrier counts are in `STATUS.md`
("Shared-memory reading of the kernel", "Specification text the shader reading rests on", and the row on the
one-full-subgroup check under "What I took from the 4070 Ti fused port"). In short: every slot of `Psh`, `Rsh`
and `Dsh` has exactly one writing invocation or is written by a cooperative-matrix store of the whole subgroup;
every read of a slot another invocation wrote comes after `memoryBarrierShared(); subgroupBarrier();`; every
read phase is closed by the same pair before the slot is written again; every barrier is in control flow that
is uniform over the subgroup (the block loop bound and both early returns depend on the workgroup only, the one
conditional block is entered on `subgroupAny`). The copy passes use no shared memory.

Arithmetic: scores and the attention x V accumulation are fp32 cooperative-matrix accumulators as in the
parent's kernels; the score is rounded to fp16 once (store to `Psh`), e is computed in fp32 against the row
maximum and rounded to fp16 for the matrix multiply, the row sum and the final division are fp32. The summation
order differs from the three-kernel path, so both candidates are judged by the reference-error rule (D3).

## Known defect: F1 (owner decision 2026-10-09 15:25 UTC: option C, no release-zone change in this campaign)

**Every number of this campaign is measured on pipelines created without the full-subgroups flag (F1).** By the
owner's decision the numbers stand as measured, nothing is rebuilt, re-gated or re-timed for F1, and F1 does not
keep this campaign open. It is a known defect of the release zone, shared with every cooperative-matrix pipeline
of every device, the shipped ones and this campaign's parent included; it will be repaired once, in the release
zone, in a change of its own before any promotion pull request, with every device gated and timed again under it.

- The requirement (`vulkan-docs`, `refpages/latest/RuntimeSpirv.md`): "VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770
  Any pipeline containing a shader with OpTypeCooperativeMatrixKHR or OpCooperativeMatrix*KHR instructions must be
  created with the VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be
  version 1.6 or greater". The flag (`refpages/latest/VkPipelineShaderStageCreateFlagBits.md`) "specifies that the
  subgroup sizes must be launched with all invocations active in the task, mesh, or compute stage".
- SPIR-V version: the two fused kernels of `orin-fused1`, the two-pass kernel of `orin-fused2` and the parent's
  attention kernels are SPIR-V 1.3 (second header word `0x00010300`, read from the `.spv` files of both shader
  compilers). SPIR-V 1.6 is not available from the dev zone: the instance is created for Vulkan 1.1
  (`vk_api/Runtime.cpp:91`, `VK_API_VERSION_1_1`).
- The pipeline code: `vk_api/Pipeline.cpp` creates every compute stage with `flags` `0u` (line 305 in the single
  pipeline path, line 540 in the batch path). It does chain the required subgroup size
  (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`, lines 291 to 299 and 527 to 534), which is why these
  kernels get subgroups of 32; it never asks for full subgroups. The runtime already reads the device feature
  (`vk_api/Device.cpp:317`, `computeFullSubgroups`); whether this device reports it is not recorded by this
  campaign (`vk-caps` prints sizes and stages, not the feature).
- What the kernel's run-time check does: `sarc_dev_orin_sdpa_fused3sb` returns before any shared-memory access
  unless `gl_NumSubgroups == 1` and `gl_SubgroupSize == 32`, so with a driver that splits a workgroup of 32
  into several subgroups the kernel writes nothing and the correctness tiers fail instead of two subgroups
  exchanging through one `Psh` / `Rsh` / `Dsh`. On this driver it never fired: 222 of 222 cases per gate are
  served by the fused kernel and correct.
- What it does not do: it does not make the pipeline valid as created; it does not guarantee that all 32
  invocations of the one subgroup are active (that is what the flag specifies); and it covers the fused kernel
  only: the parent's QK^T, softmax and attention x V kernels, which still serve decode and unaligned prompts in
  the final stack, and the shipped cooperative-matrix linear kernels have no such check.
- The two forms of the repair, as they were put to the owner (the B580 campaign's wording; input to the later
  release-zone change, neither is authorised here). **Form A**, per shader, inert unless a shader asks for it: a
  yaml parameter that `gen_vulkan_spv.py` passes into `ShaderInfo` beside the required subgroup size, a field in
  the pipeline descriptor and its hash / equality, and in `Pipeline.cpp` the stage flag when the field is set, a
  required subgroup size is in force and the device reports `computeFullSubgroups`; files `gen_vulkan_spv.py`,
  `vk_api/Shader.{h,cpp}`, `vk_api/Pipeline.{h,cpp}` and where the descriptor is filled, an estimated 30 to 40
  lines; it leaves the parent's and the shipped pipelines with the defect. **Form B**, for every pipeline with a
  required subgroup size: in `Pipeline.cpp` at both places, set the flag whenever a required subgroup size is in
  force and the device reports `computeFullSubgroups`, about 10 lines; it repairs the shipped and the parent's
  pipelines too and therefore changes how every such pipeline of every device is created (with the flag the
  local size in X must be a multiple of the required subgroup size for each of them), so every device's results
  need gating and timing again.
- Decode with the fused node present (the owner's item 5; one 31-token decode run per `verify.sh`, 1B only, not
  a timed session and not investigated): parent environment 19.08 / 10.89 tok/s (4w / 8da4w, `s0-parent-verify`);
  `orin-fused1` 18.91 / 10.84 (`s3-c1`) and 18.89 / 10.86 (`s5-c1`), that is 0.9 to 1.0 % and 0.3 to 0.5 % lower;
  `orin-fused2` 18.98 / 10.83 (`s6-c2`). With nothing selected: 19.15 / 10.95 on the parent build, 19.18 / 10.94 on
  `topic4`. Single runs; the differences are of the size the B70 confirmation saw (0 to 2.5 %).

## Known limitation: shader compiler of the cross build (owner decision 2026-10-09 15:50 UTC)

**Every number of this campaign is measured on builds whose shaders were compiled by the cross image's `glslc`,
not by the compiler the goldens were made with.** By the owner's decision byte identity of the measured build's
SPIR-V with `sarc/golden/spirv.json` is not required for this campaign, the numbers stand as measured on `topic4`,
nothing is measured again, and the difference does not keep the campaign open.

- The two compilers: the cross image `localhost/et-jetson-cross:jp7.2.1` has shaderc v2026.1; the goldens were
  made with the `glslc` of `localhost/et-vk-build:rocky10` (shaderc v2023.8, spirv-tools v2025.4, glslang
  `73743588`; `.artifacts/pinned-glslc/SOURCE.txt`).
- `sarc/tools/spirv_golden.py`, unmodified, reads `FAIL (53 shipped variants)` with 14 DIFF lines on the measured
  builds `parent` and `topic4` (the same 14; `.artifacts/build/{parent,topic4}.golden.txt`). All 14 are shipped
  variants of other devices: 4 of the Radeon 780M (two 8da4w linear, its QK^T and attention x V), 2 of the Arc
  B580 / B70, 2 of the Xclipse M51, 3 of the Adreno 840, 3 of the Mali G1. None is a kernel the Orin rows
  dispatch. Between the parent build and `topic4` all 53 shipped variants are byte-identical (`tools/shipped.py`).
- This campaign's own kernels are compiled to other bytes than the pinned compiler produces. All 1620 shaders
  of `topic4`'s source were compiled with each compiler by the tree's `gen_vulkan_spv.py`
  (`.artifacts/shadercheck/pinned-topic4/`): the cross image's output equals the measured build's in all 1620,
  the pinned compiler's passes the golden, and 348 files differ between the two (248 named `sarc_*`, of them 9 dev-zone ones with the 4 `sarc_dev_orin_*` fused kernels, and the 6
  stock `sdpa_compute_attn_weights_*`). The comparison that exists is
  byte identity and file size; no instruction-level comparison was made:

  | kernel | role | cross image, bytes | pinned compiler, bytes |
  |---|---|---:|---:|
  | `sarc_dev_orin_sdpa_fused3sb_d64_t32x32g11s32rko` | candidate 1, head_dim 64 | 16524 | 16684 |
  | `sarc_dev_orin_sdpa_fused3sb_d128_t16x64g11s32rko` | candidate 1 and 2, head_dim 128 | 20832 | 21472 |
  | `sarc_dev_orin_sdpa_fused3sb_d64_t32x32g11s32rk` | candidate 2, head_dim 64 | 19240 | 19400 |
  | `sarc_sdpa_qk_coopmat_4070ti_pk_t128x64k64g42s32nf` | parent QK^T | 10200 | 10216 |
  | `sarc_sdpa_av_coopmat_4070ti_t64x64k32g42s32` | parent attention x V | 6616 | 6632 |
  | `sarc_sdpa_attn_weights_softmax_buffer_half_orin_g64` | parent softmax | 10172 | 10172 (identical) |
  | `sdpa_compute_attn_weights_tiled_buffer_buffer_half` | stock attention (pristine arm) | 11236 | 11256 |

  So both arms of every comparison of this campaign were compiled by the same compiler, and it is not the
  pinned one. What the pinned compiler's kernels would time or compute on this device is not measured.
- The first Orin campaign (`sarc-1.5-orin-prefill-refine`) used the same image: its builds read the same FAIL,
  and this campaign's `parent` build is byte-identical to its final build `topic14` in all 1610 shaders.
- What exists and is not measured: builds `parentg` (`8973ced76`) and `topic4p` (`0bed38090`), the same commits
  compiled with the pinned compiler inside the cross recipe (`GLSLC_DIR`, off by default): `spirv golden: PASS
  (53 shipped variants)` on both. They stay as evidence of what the pinned compiler produces. `topic4g` failed
  to build (rc 141, a logging line of mine) and is nothing. The device queue `chain10`, which was to gate and
  time everything again on them, was stopped at 15:48:46 UTC inside its first step (the `verify.sh` of
  `s0g-parent-verify`, about 14 minutes in); that step's partial output is kept on the device under
  `stage/superseded/chain10-stopped-owner-decision-1550/` and is evidence for nothing.

## Hook control (D4 conditions)

| condition | evidence |
|---|---|
| with nothing selected every dispatch is as before | `s7n-noenv`: unmodified `verify.sh` on `topic4` (`0bed38090`: the hook and the whole dev zone of this campaign) with no environment equals `s0n-noenv` (the parent build, no environment) line by line, rates aside (34 lines, 0 differing; `gate_check.py verify` ACCEPT, 0 findings; no `[sarc_dev]` banner). The same on `topic1` (`s2n-noenv`) |
| the same under the parent's own environment | A/A `s4-aa2` (parent build against `topic4`, both `orin-refine5` + `orin_g64`): geomean +0.07 %, next token SAME in 24 of 24 rows |
| `test_sarc_select` unchanged | release tables: `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)`, the parent's line; with the dev zone: 1593 checks, 33 rows |
| shipped SPIR-V unchanged | 53 of 53 shipped variants byte-identical between the parent build and `topic4` (`tools/shipped.py`). `spirv_golden.py` itself reads FAIL with 14 DIFF lines on every cross build including the parent's and the first campaign's: the cross image's glslc (shaderc v2026.1) is not the one the goldens were made with; none of the 14 is a kernel the Orin rows dispatch. Of the 1620 SPIR-V files of `topic4`, 1610 are byte-identical to the parent build's 1610 (the 8da4w kernel shared with the 4070 Ti among them) and 10 are new |
| reproducible from the committed branch, no local patch | `topic4` is an export of commit `0bed38090` of this branch; the gate `s5-c1` and both final timed sessions are on that build; the branch head differs from it only under this change directory |
| `sarc/tools/check.sh --no-build` | PASS (output in `STATUS.md`). It does not report the hook: `impl/sarc/` is inside the zones it checks and `impl/SDPA.cpp` is listed in `sarc/HOOKS`; the four files are named here under the owner decision of 2026-10-05 |

## Result

Final stack: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1` on build `topic4`. Tok/s, 2048-token
prefill, median of 5 valid interleaved runs per arm:

| cell | published `dev/1.5` | pristine (`s8-pristine`) | tuned parent (`s5-c1`) | final stack (`s5-c1`) | over the parent | over pristine (`s8-pristine`) |
|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | 890.82 | 891.60 | 1490.54 | 1656.96 | +11.17 % | +85.24 % |
| 1B 8da4w | 822.82 | 823.81 | 1379.12 | 1520.42 | +10.25 % | +84.70 % |
| 3B 4w | 360.37 | 360.50 | 629.19 | 659.58 | +4.83 % | +82.96 % |
| 3B 8da4w | 320.30 | 320.35 | 570.32 | 595.35 | +4.39 % | +85.79 % |
| 8B 4w | 189.74 | 189.84 | 295.44 | 305.49 | +3.40 % | +60.94 % |
| 8B 8da4w | 170.43 | 170.48 | 269.05 | 277.24 | +3.05 % | +62.60 % |
| geomean | | | | | **+6.13 %** | **+76.70 %** |

Both sessions: 60 of 60 timed runs valid, GPU clock 612 MHz in every run, thermal throttle state 0 in every
clock sample, repeat spread at most 0.54 %, next token the same as the other arm on all four prompts in every
cell. A/A under the same rules (`s4-aa2`): +0.07 %.

| candidate | profile | gate | gain over its parent | outcome |
|---|---|---|---:|---|
| 1: fused attention kernel, one pass, packed K / V | `orin-fused1` | `s5-c1` on `topic4` (first: `s3-c1` on `topic1`, +6.01 %) | +6.13 % | GATE_ACCEPTED, no next-token item differs; arithmetic change, so the reference-error evidence is recorded beside it (criterion 1 met on the five S = 2048 cases; 41-prompt probe: no gross divergence) |
| 2: two-pass form for head_dim 64 | `orin-fused2` | `s6-c2` on `topic4`, against candidate 1 | +0.54 % (1B +1.72 / +1.59 %) | GATE_ACCEPTED on correctness; every cell inside the +-2 % band: not adopted. Gated as candidate 2 by the owner's decision of 2026-10-09 01:00 UTC; the pre-registered clause of `thresholds.txt` (an unpacked form) stays unedited and by its letter there was no candidate 2 |

Stop: two candidates at most (task), candidate 2 sub-threshold; closed by N3 with candidate 1 as the result.

Where the gain comes from (warm ETDump of `s5-c1`, ms per 2048-token prefill, parent -> final):

| cell | QK^T + softmax + attn*V | fused kernel + K / V copy | attention | dispatch total |
|---|---:|---:|---|---|
| 1B 4w | 302.1 | 172.2 + 3.1 | 302.1 -> 175.3 (-42 %) | 1359.2 -> 1225.2 |
| 1B 8da4w | 302.1 | 171.9 + 3.1 | 302.1 -> 175.0 | 1469.9 -> 1333.9 |
| 3B 4w | 516.0 | 371.0 + 11.5 | 516.0 -> 382.5 (-26 %) | 3236.6 -> 3086.8 |
| 3B 8da4w | 516.6 | 371.2 + 11.3 | 516.6 -> 382.4 | 3571.0 -> 3420.0 |
| 8B 4w | 783.2 | 561.8 + 13.0 | 783.2 -> 574.7 (-27 %) | 6916.6 -> 6686.1 |
| 8B 8da4w | 783.3 | 561.1 + 12.9 | 783.3 -> 573.9 | 7588.6 -> 7363.1 |

Attention is 89 to 95 % of the reduction of the dispatch total (95 and 93 % on 1B, 89 % on 3B, 90 and 93 % on 8B). The rest of it, from the raw sums of the same
traces (parent -> final, six cells): linear GEMM -0.1 to -0.5 % (655.2 -> 654.5 ms on 1B 4w, 5304.2 -> 5297.0 on
8B 8da4w); copy / view / other -0.2 to -0.7 % (154.3 -> 153.3 ms on 1B 8da4w); everything that is none of the
three (elementwise, RMSNorm, RoPE, quantize, ...) -0.8 to -4.6 %, that is 4 to 13 ms per prefill (135.4 -> 129.9 ms
on 1B 4w, 277.3 -> 264.6 on 3B 4w). I have not located which kernels of that remainder got faster.

Against the freshly measured roofs (igpu-roofline `fast`, 2026-10-09, driver 595.78, clocks as found;
`results/orin/roofline/2026-10-09-fast/`: matrix fp16 9.719 TFLOP/s, fp16 -> fp32 9.735, int8 19.517 TOP/s): 4w
linear 62.7 / 63.8 / 63.2 % (1B / 3B / 8B), 8da4w linear 26.4 / 27.3 / 27.7 %, both as the parent; the fused
attention kernel 1.60 / 1.95 / 1.96 TFLOP/s = 16.4 / 20.0 / 20.1 % of the fp16 -> fp32 roof.

Comparison with the 4070 Ti port of the same kernel (`origin/topic/4070ti-fused-port`):

| | RTX 4070 Ti SUPER | Jetson Orin Nano |
|---|---|---|
| geomean over the tuned parent | +11.64 % | +6.13 % |
| per cell (1B / 3B / 8B, 4w and 8da4w) | +19 / +21, +8 / +10, +6 / +7 % | +11.2 / +10.2, +4.8 / +4.4, +3.4 / +3.0 % |
| attention time removed | 66 % (1B), 47 to 49 % (3B, 8B) | 42 % (1B), 26 to 27 % (3B, 8B) |
| fused kernel, share of the matrix roof | 53 to 63 % | 16 to 20 % |
| K / V copy pass per prefill | 0.1 to 0.4 ms | 3 to 13 ms |
| error against the fp32 reference, 1B S = 2048 (rms, maximum) | 2.049e-5, 7.23e-4 | 2.049e-05, 7.227e-04 |
| faster form for head_dim 64 | one pass (the only one built) | two passes (-12.5 % kernel time, +1.7 % end to end, not adopted) |

Same kernel, same arithmetic (the reference errors agree to the printed digits), half the gain: this device
feeds its matrix unit from DRAM far slower than from shared memory, and the kernel loads K and V tiles straight
from DRAM.

Memory of the K / V copies (`results/orin/mem1/rows.csv`): two fp16 buffers of 8 x head_dim x 2560 elements,
5.2 MB a pair for 1B and 10.5 MB for 3B and 8B. Largest drop of available memory during a prefill run, parent
against final stack: 935 / 965 MB (1B 4w), 885 / 881 (1B 8da4w), 1976 / 1974 and 1982 / 1984 (3B), 4327 / 4344 and
4577 / 4579 (8B): no measurable cost. The 8B runs swap out a few MB in both arms alike.

Negative results: the unpacked forms are 1.5 (head_dim 64) and 3.4 times (128) slower per layer; the two-pass
form is 38 % slower for head_dim 128; candidate 2 is inside the band (`STATUS.md`, "Negative results").

What limits further progress: the fused kernel's 16 to 20 % of the roof (a kernel that stages K and V tiles
through shared memory is what this device would need: a new kernel, not a port); linear GEMM is 53 to 72 % of the
prefill and where the first campaign left it; copy / view / other is 8 to 22 %.

## Final verification

`STATUS.md`, "Final verification (R11)": every item on build `topic4`, whose commit is the last that changes code.
One item is not passed as the rules write it and is accepted by the owner: `spirv_golden.py` on the measured build
("Known limitation: shader compiler of the cross build"). F1 is the other owner-accepted label ("Known defect: F1").
What was taken from the 4070 Ti port's review and changed the course of this campaign: the thermal-throttle
record in every timed run, the test change as insert-only blocks, the 12 passes of tier `all` (already present)
and the kernel's own one-full-subgroup check; the sessions measured before them (`s1-aa`, `s3-c1` on `topic1`)
are kept as evidence and are not the reported result.
