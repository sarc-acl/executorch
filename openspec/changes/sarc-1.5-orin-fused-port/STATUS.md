# STATUS: sarc-1.5-orin-fused-port

**2026-10-09 16:06 UTC. Review round 2 answered (wording, acceptance labels; one item is the coordinator's, below). Result: `orin-fused1`, +6.13 % geomean over the tuned
parent (`s5-c1`) and +76.70 % over pristine `dev/1.5` (`s8-pristine`), measured on build `topic4`; candidate 2
(`orin-fused2`) +0.54 %, gated, not adopted. Two labels go with every number, both by owner decision and neither
keeping the campaign open: the shaders were compiled by the cross image's `glslc`, so the measured builds fail
`spirv_golden.py` on 14 shipped variants of other devices (section "Known limitation", decision of 15:50 UTC),
and the pipelines were created without the full-subgroups flag (section "Known defect: F1", decision of
15:25 UTC). The re-measurement on the pinned-compiler builds that the entry of 15:35 UTC announced was stopped
by the owner (`chain10`, killed 15:48:46 UTC in its first step) and produced no result.**

**For the coordinator (blocks a clean `check.sh`):** the review tooling puts `.agents/skills/review-notes/SKILL.md`
into this working copy, untracked. `sarc/tools/check.sh --no-build` fails its zone rule on it and on nothing else.
Please place that skill outside the checkout and run the unmodified check again; the campaign will not edit
`check.sh` or `sarc/HOOKS`, commit the file, or delete a file it does not own.

All times are UTC from `date -u`.

## Running now

- Workstation: nothing of this campaign (no build, no queue).
- Device: nothing. `chain10` was ended with `dkill.sh` at 15:48:46 UTC (owner decision of 15:50 UTC); no
  `llama_main`, `test_llama_microbench` or `logits_dump` of this campaign is left, 5.8 GB available, GPU at
  306 MHz and 57 C afterwards. No `ABORTED`, `GPU_GONE` or `HOLD` file.
- An older hmz run of this same campaign (`hmz exec` pid 8272, started 01:20 UTC, its actor session waiting on
  `chain10` since 15:44 UTC) was still alive when this session started at 15:47 UTC; it had ended by itself by
  15:51 UTC. I did not touch it, and it started nothing after `chain10` was stopped (device checked again).

## Review of 15:00 UTC: what it found and what I did

| finding | what I did |
|---|---|
| `spirv_golden.py` FAILS on `topic4` and on `parent` (14 DIFF lines, the same set); `shipped.py`'s parent comparison cannot replace it | **Settled by the owner at 15:50 UTC: byte identity with the golden is not required for this campaign, no re-measurement (section "Known limitation" below).** What I had done before that decision: checks made first, no device time: (1) `check.sh` with its build steps (release export, host build in `localhost/et-vk-build:rocky10`): `host build OK`, `spirv golden: PASS (53 shipped variants)`. (2) All 1620 shaders of `topic4`'s source compiled with each compiler by the tree's `gen_vulkan_spv.py`: the cross image's output equals the measured build's in all 1620 (the method is right), the pinned one passes the golden, and 348 differ between the two (239 `sarc_*`, the stock `sdpa_compute_*`, 4 `sarc_dev_orin_*` among them). (3) The golden's `glslc` with its two SPIRV-Tools libraries runs inside the cross image and gives all 1620 shaders byte-identical to the pinned image's. So the recipe copy got an option (`GLSLC_DIR`, off by default; `tools/jetson-cross/{container,build}.sh`, `tools/build-orin.sh`; the compiler is kept under `.artifacts/pinned-glslc/` with a manifest) and the two commits were built again with it (`parentg`, `topic4p`; kept, not measured). No golden, shipped shader, tolerance or `sarc/tools` file is touched |
| `proposal.md`: "everything else is unchanged within 0.3 %" is false | Corrected from the raw sums: attention is 89 to 95 % of the reduction; linear GEMM -0.1 to -0.5 %, copy / view / other -0.2 to -0.7 %, everything else -0.8 to -4.6 % (4 to 13 ms per prefill), not located |
| the residual row of "Where the time goes now" used rounded subtraction | Recomputed from the raw sums (1B: 130 / 231 ms) |
| fp16 roof 9.7195 rounded to 9.720 | 9.719 |
| `check.sh` not re-run by the reviewer | Re-run with and without the build steps (end of this file). **It now prints FAIL at step 1** for one untracked file that is not mine and that I have not touched: `.agents/skills/review-notes/SKILL.md` (dated 2026-10-04, in the working copy since about 14:46 UTC; a review-instruction file of the review tooling). Every other step passes. It is not committed and not part of this branch; whoever placed it should move it out of the working copy or the check keeps failing on it. Still so at 16:06 UTC (the file was absent for a few minutes around 15:50 UTC and came back); see the end of this file |

Build tag `topic4g` failed (rc 141: a `glslc --version | head -1` line I had added to the recipe, killed by
`pipefail`; `parentg` got through the same line by timing) and is not used; `topic4p` is the same commit with
that line fixed. The recipe hash of `parentg` and `topic4p` therefore differs in that one logging line.

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

## Result

Tok/s, 2048-token prefill, median of 5 valid interleaved runs per arm; every number recomputed from `runs.csv`.

| cell | published `dev/1.5` (`cells.csv`) | pristine, re-measured (`s8-pristine`) | tuned parent (`s5-c1`) | final stack (`s5-c1`) | over the parent | over pristine (`s8-pristine`) |
|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | 890.82 | 891.60 | 1490.54 | 1656.96 | +11.17 % | +85.24 % |
| 1B 8da4w | 822.82 | 823.81 | 1379.12 | 1520.42 | +10.25 % | +84.70 % |
| 3B 4w | 360.37 | 360.50 | 629.19 | 659.58 | +4.83 % | +82.96 % |
| 3B 8da4w | 320.30 | 320.35 | 570.32 | 595.35 | +4.39 % | +85.79 % |
| 8B 4w | 189.74 | 189.84 | 295.44 | 305.49 | +3.40 % | +60.94 % |
| 8B 8da4w | 170.43 | 170.48 | 269.05 | 277.24 | +3.05 % | +62.60 % |
| geomean | | | | | **+6.13 %** | **+76.70 %** |

The "over pristine" column is the ratio measured inside `s8-pristine` (its final-stack medians are 1651.61,
1521.55, 659.58, 595.18, 305.54, 277.21, within 0.3 % of the `s5-c1` ones). The task expected +5 to +12 %: the
result is inside the band, at its lower end; the 4070 Ti port of the same kernel closed at +11.64 %.

## Final verification (R11), all on build `topic4` (cross image's `glslc`: "Known limitation" above)

| item | session | result |
|---|---|---|
| the build is the branch head's code | | `topic4` is an export of `0bed38090`; `git diff --name-only 0bed38090 HEAD` lists nothing outside `openspec/changes/sarc-1.5-orin-fused-port/` |
| unmodified `verify.sh`, final environment, on the timed binaries | `s5-c1` | 34 lines equal to `s0-parent-verify` with the rates removed (0 differing); 22 of 22 runner calls rc 0; `gate_check.py verify` ACCEPT |
| attention tiers | `s5-c1` | `all`, `extended`, `full` x 12 and `peaked`, `fused` x 3: 222 of 222 cases PASSED, 0 mismatches, fused kernel alone, `pairing=ok` |
| shipped SPIR-V | build | **`spirv_golden.py`: FAIL, 14 DIFF lines** (the same 14 on the parent build; cross image's glslc). 53 of 53 shipped variants byte-identical to the parent build; none of the 14 is a shipped Orin kernel. Not passed as R5 / R7 write it; byte identity with the golden is not required for this campaign (owner decision 2026-10-09 15:50 UTC, "Known limitation") |
| reference error, criterion 1 | `sdpa-error2` | rms and maximum not larger than the parent's on all five S = 2048 cases |
| real-text evidence | `probe/final-fused/` | no differing next-token item; gross-divergence check not met in any cell; `ref_error_rule.py` MET |
| how candidate 1 is recorded | | `ACCEPTED (reference-error rule, owner decision 2026-10-04)`; `results/orin/sdpa-error2/`, `results/orin/probe/final-fused/`; differing next-token items: none |
| timed session against the tuned parent | `s5-c1` | +6.13 % geomean, 60 of 60 valid, throttle state 0 |
| timed session against the pristine state | `s8-pristine` | +76.70 % geomean, 60 of 60 valid, throttle state 0 |
| hook control, nothing selected (D4) | `s7n-noenv` | `verify.out` equal to `s0n-noenv` line by line; no `[sarc_dev]` banner |
| A/A under the final rules | `s4-aa2` | +0.07 % geomean |
| `sarc/tools/check.sh --no-build` | workstation | at the end of this file |

## Real-text probe of the final stack on `topic4` (`chain9`, 11:18 to 13:27 UTC; `results/orin/probe/final-fused/`)

41 prompts x 6 cells, default and tiled linear, `topic4`'s `logits_dump` with `orin-fused1`, against the parent's
two arms of `chain4`. **The logits of both arms are bit-identical to those `topic1`'s kernel produced** (sha256
over the six logits files of an arm: `4d8319f8...` default, `49d92548...` tiled, the same for `c1-*` and
`final-*`, written 8 hours apart by different binaries): the one-full-subgroup check changes no arithmetic, and
the kernel is deterministic over 246 prompt x cell evaluations per arm. `compare.csv` therefore equals the
`c1-fused` table below in every digit: `differing-items.txt` empty, largest mean KL 0.041 nat (1B 8da4w), top-1
differs on at most 2 of 41 prompts, `ref_error_rule.py` verdict MET with `sdpa-error2`'s files.

## Memory of the K / V copies (`mem1`, 13:27 to 13:58 UTC, `topic4`; `results/orin/mem1/rows.csv`)

The fused node adds tile-packed copies of K and V: two fp16 buffers of kv_heads x head_dim x context capacity =
8 x 64 x 2560 x 2 B = 2.6 MB each for 1B and 8 x 128 x 2560 x 2 B = 5.2 MB each for 3B and 8B (5.2 and 10.5 MB a
pair). Largest drop of `MemAvailable` during one prefill run (sampled every 0.2 s), parent environment against
`orin-fused1`, round 2 (round 1 of the parent arm starts with more free memory because it is the first run after
the model change, so its drop is not comparable):

| cell | parent, MB | `orin-fused1`, MB | difference | lowest `MemAvailable`, parent / fused |
|---|---:|---:|---:|---|
| 1B 4w | 935 | 965 | +30 | 4913 / 4883 |
| 1B 8da4w | 885 | 881 | -4 | 4958 / 4962 |
| 3B 4w | 1976 | 1974 | -2 | 3876 / 3875 |
| 3B 8da4w | 1982 | 1984 | +2 | 3869 / 3863 |
| 8B 4w | 4327 | 4344 | +17 | 1503 / 1509 |
| 8B 8da4w | 4577 | 4579 | +2 | 1276 / 1275 |

No measurable cost: within +30 / -4 MB, the size of the run-to-run variation. A pair per layer would be 84 MB
(1B), 293 MB (3B) and 335 MB (8B) and would show; it does not, so the copies are temporaries the graph shares
between layers (my reading of the numbers, not checked in the allocator). The copies fit; the 8B model leaves
1.3 to 1.5 GB available at its lowest in both arms. Swap-out during the probe: 0 to 16 pages per run in round 2
for either arm; in the timed sessions 8B runs swap out up to 11 MB per run in the parent arm and up to 5 MB in
the candidate arm (above, `s5-c1`): the device swaps a little on 8B with or without the fused node.

## Fresh roofs (`roof-final`, 13:59 to 14:41 UTC; `results/orin/roofline/2026-10-09-fast/`)

igpu-roofline `fast` plan, tree `fleet-quick-20260925` (source commit `3bd916f6`), driver 595.78, clocks as found
(`nvhost_podgov`, 306 to 612 MHz, power mode 15 W; nothing pinned), status finished. This is the run every
percentage below uses.

| roof | 2026-10-09 | first campaign, 2026-10-05 |
|---|---:|---:|
| matrix fp16 16 x 16 x 16 | 9.719 TFLOP/s | 9.716 |
| matrix fp16 -> fp32 | 9.735 TFLOP/s | 9.722 |
| matrix int8 16 x 16 x 32 | 19.517 TOP/s | 19.482 |
| the same fed from shared memory (fp16 / fp16 -> fp32 / int8) | 9.516 / 8.962 / 17.853 | 9.511 / 8.771 / 17.818 |
| DRAM read / write / copy | 62.3 / 58.1 / 64.0 GB/s | 62.1 / 58.0 / 64.1 |

Final stack against them (warm ETDump of `s5-c1`; `tools/roof_util.py`):

| | 1B | 3B | 8B |
|---|---|---|---|
| 4w linear, share of the fp16 matrix roof | 6.09 TFLOP/s = 62.7 % | 6.21 = 63.8 % | 6.15 = 63.2 % |
| 8da4w linear, share of the int8 matrix roof | 5.15 TOP/s = 26.4 % | 5.33 = 27.3 % | 5.40 = 27.7 % |
| fused attention kernel, per layer | 17.2 GFLOP in 10.76 ms = 1.60 TFLOP/s = 16.4 % of the fp16 -> fp32 roof | 25.8 GFLOP in 13.25 ms = 1.95 = 20.0 % | 34.4 GFLOP in 17.56 ms = 1.96 = 20.1 % |

(The attention work is counted as the causal half of QK^T and of attention x V: 2 x S x S x head_dim x heads
flops per layer at S = 2048. At the roof a layer would take 1.8 / 2.6 / 3.5 ms.) The linear kernels are the
parent's and sit where the first campaign left them.

## Where the time goes now, and what limits further progress

Final stack, warm ETDump, ms per 2048-token prefill (`s5-c1`, candidate arm; each row rounded from the raw sum
of `attention.csv`, the last one being total - GEMM - attention - copy):

| | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 655 | 774 | 1860 | 2165 | 4652 | 5297 |
| attention (fused kernel + K / V copy) | 175 | 175 | 382 | 382 | 575 | 574 |
| copy / view / other | 266 | 153 | 579 | 350 | 983 | 572 |
| everything else (elementwise, RMSNorm, RoPE, quantize, ...) | 130 | 231 | 265 | 523 | 477 | 920 |
| total | 1225 | 1334 | 3087 | 3420 | 6686 | 7363 |
| attention share | 14.3 % | 13.1 % | 12.4 % | 11.2 % | 8.6 % | 7.8 % |
| gain if attention cost nothing | +16.7 % | +15.1 % | +14.1 % | +12.6 % | +9.4 % | +8.5 % |

1. **The fused kernel runs at 16 to 20 % of the matrix roof here**, against 53 to 63 % on the 4070 Ti and 71 % on
   the 780M. It loads every K and V tile straight from DRAM (its design: no shared staging), and the first
   campaign measured that this device feeds its matrix unit far slower from DRAM than from shared memory (2.8
   against 8.8 TFLOP/s in its run). That is why the same port gives +6.1 % here and +11.6 % on the 4070 Ti, and
   why attention still is 8 to 14 % of the prefill. A fused kernel that stages K and V tiles through shared
   memory is the kernel this device would need; it is a new kernel, outside a port campaign.
2. **For head_dim 64 the two-pass form is the faster kernel on this device** (9.4 against 10.8 ms a layer): it
   declares 2 KB less shared memory per workgroup and has no rescale branch. Gated and correct (`orin-fused2`),
   +1.7 % on 1B, inside the band.
3. Linear GEMM is 53 to 72 % of the prefill and unchanged by this campaign: 4w at 63 % of its roof, 8da4w at 27 %.
4. Copy / view / other is 8 to 22 % of the prefill, untouched.

## Negative results (kept with their numbers)

| what | where | number | outcome |
|---|---|---|---|
| unpacked forms of the kernel (`ro`, `r`: no K / V copy pass) | `sdpa-screen1` | 1.5 times (head_dim 64) and 3.4 times (128) slower per layer than the packed forms | no "unpacked" candidate; the copy pass (3 to 13 ms per prefill) saves far more than it costs |
| two-pass form for head_dim 128 | `sdpa-screen1` | 19.2 against 13.9 ms (3B), 25.3 against 18.3 ms (8B) | one pass stays |
| candidate 2, two-pass form for head_dim 64 | `s6-c2` | +0.54 % geomean (1B +1.72 / +1.59 %) | gate accepted, inside the band, not adopted |
| the first form of the microbench change (9 existing lines edited) and the sessions without a throttle record | `s1-aa`, `s3-c1` on `topic1` | +6.01 % | kept as measured, superseded by `s4-aa2` and `s5-c1` on `topic4` |

## Final stack against the pristine state: `s8-pristine` (10:27 to 11:18 UTC), `SESSION_ACCEPTED`

Parent arm: build `parent` (`8973ced76`) with no environment (the `dev/1.5` state of this device). Candidate arm:
build `topic4` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`. Recomputed from `runs.csv`:

| cell | pristine | published (`cells.csv`) | final stack | gain over pristine | repeat spread (pristine / final) |
|---|---:|---:|---:|---:|---|
| 1B 4w | 891.60 | 890.82 | 1651.61 | +85.24 % | 0.26 / 0.08 % |
| 1B 8da4w | 823.81 | 822.82 | 1521.55 | +84.70 % | 0.32 / 0.37 % |
| 3B 4w | 360.50 | 360.37 | 659.58 | +82.96 % | 0.09 / 0.10 % |
| 3B 8da4w | 320.35 | 320.30 | 595.18 | +85.79 % | 0.05 / 0.06 % |
| 8B 4w | 189.84 | 189.74 | 305.54 | +60.94 % | 0.07 / 0.07 % |
| 8B 8da4w | 170.48 | 170.43 | 277.21 | +62.60 % | 0.08 / 0.08 % |

Geomean **+76.70 %** over the pristine state (the first campaign closed at +66.65 %; 1.6665 x 1.0613 = 1.769). 60
of 60 timed runs valid: clock 612 MHz in every run of both arms (the pristine arm does not settle lower here),
throttle state 0 in all 9647 clock samples, 11 to 114 samples per window, start temperature 53 to 58 C. Next
token pristine vs final SAME in 24 of 24 rows. `gate_check.py session` and `env`: ACCEPT, 0 findings.
Attention per prefill, pristine -> final (warm ETDump): 1215 -> 175 ms (1B), 2913 -> 382 ms (3B), 4436 -> 574 ms (8B).

## Hook control on `topic4`: `s7n-noenv` (10:09 to 10:27 UTC), `GATE_ACCEPTED`, owner decision D4

Unmodified `verify.sh` on build `topic4` (`0bed38090`: the hook + the whole dev zone of this campaign) with
nothing selected, against `s0n-noenv` (the parent build, nothing selected): `verify.out` (34 lines) equal line by
line with the rates removed, **0 differing lines**; `gate_check.py verify`: ACCEPT, 0 findings; 22 of 22 runner
calls rc 0; no `[sarc_dev]` banner in any log; default-arm prefill 890.05 / 823.15, 360.37 / 320.30,
189.79 / 170.43 tok/s. Shipped SPIR-V of `topic4`: 53 of 53 byte-identical to the parent build. `test_sarc_select`
on the release tables: in the `check.sh` output at the end of this file.

## Candidate 2 (`orin-fused2`) on `topic4`: gate `s6-c2`, `GATE_ACCEPTED 2026-10-09T10:09:39Z`, NOT ADOPTED (+0.54 %)

Both arms on build `topic4`: parent arm = candidate 1 (`orin-fused1`), candidate arm = `orin-fused2` (two-pass
packed kernel `d64_t32x32g11s32rk` for head_dim 64, the same one-pass kernel for 128). Recomputed from `runs.csv`
(`results/orin/sessions/s6-c2/`):

| cell | candidate 1 | candidate 2 | gain | outside the +-2 % band | repeat spread (c1 / c2) | next token (4 prompts) |
|---|---:|---:|---:|---|---|---|
| 1B 4w | 1651.61 | 1680.07 | +1.72 % | no | 0.32 / 0.08 % | SAME |
| 1B 8da4w | 1522.68 | 1546.83 | +1.59 % | no | 0.07 / 0.08 % | SAME |
| 3B 4w | 659.37 | 659.58 | +0.03 % | no | 0.10 / 0.03 % | SAME |
| 3B 8da4w | 595.35 | 595.00 | -0.06 % | no | 0.09 / 0.06 % | SAME |
| 8B 4w | 305.54 | 305.44 | -0.03 % | no | 0.09 / 0.10 % | SAME |
| 8B 8da4w | 277.32 | 277.36 | +0.01 % | no | 0.12 / 0.05 % | SAME |

Geomean **+0.54 %**. By the rule fixed at 01:22 UTC (adopt only at 2 % geomean or more) the final stack stays
`orin-fused1`. The two 1B cells are a real, repeatable +1.6 to +1.7 % (repeat spread 0.1 to 0.3 %), but they are
inside the band the campaign fixed before measuring, so they are reported and not claimed.

- **Scope of this acceptance: the gate and criterion 1 only, not a full D3 acceptance.** For `orin-fused2` there
  is the gate (below), the reference error on the five S = 2048 cases (`sdpa-error2`) and 24 of 24 next-token
  rows SAME. There is **no broad real-text probe of `orin-fused2`** (D3 items 2 and 3: logits, KL, perplexity on
  41 prompts): `probe/c1-fused/` and `probe/final-fused/` are both `orin-fused1`. The candidate is not adopted, so
  no D3 acceptance is claimed for it and none was measured; whoever adopts it later owes that probe first.
- The gate itself passed: 60 of 60 timed runs valid (throttle state 0 in all 7953 clock samples, clock 612 MHz,
  start temperature 54 to 58 C); SDPA 222 of 222 cases PASSED with 0 mismatches, 126 served by the two-pass d64
  kernel and 96 by the one-pass d128 kernel, `pairing=ok` on all; `verify.out` equal to the parent snapshot line
  by line (0 differing lines); the four `gate_check.py` steps ACCEPT with 0 findings; next token SAME in 24 of
  24 rows; reference-error criterion 1 met (`sdpa-error2`, below). Pre-check `c2-pre`: 26 of 26 cases PASSED.
- Where the 1B gain comes from (warm ETDump, `sessions/s6-c2/trace/attention.csv`): the fused kernel 172.5 ->
  151.0 ms per prefill on 1B 4w and 171.8 -> 150.8 ms on 8da4w (-12.5 %; the kernel screen had 11.6 -> 9.6 ms per
  layer, -17 %); 3B and 8B unchanged (371 and 561 ms), as they run the same kernel. 21.5 ms of a 1225 ms prefill
  is the +1.7 %.
- For whoever takes this further: the two-pass form is the better kernel for head_dim 64 on this device and is
  built, gated and selectable (`orin-fused2`); it just does not clear the campaign's own bar.

## Candidate 1 on `topic4`: gate `s5-c1`, `GATE_ACCEPTED 2026-10-09T08:00:47Z all steps passed`

Parent arm: build `parent` (`8973ced76`) with the parent environment. Candidate arm: build `topic4` (`0bed38090`)
with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`. Tok/s, median of 5 valid interleaved runs per
arm, recomputed from `runs.csv` with a script of my own (`results/orin/sessions/s5-c1/`):

| cell | parent | candidate 1 | gain | repeat spread (parent / candidate) | `s3-c1` gain (`topic1`) | next token (4 prompts) |
|---|---:|---:|---:|---|---:|---|
| 1B 4w | 1490.54 | 1656.96 | +11.17 % | 0.36 / 0.40 % | +10.89 % | SAME |
| 1B 8da4w | 1379.12 | 1520.42 | +10.25 % | 0.54 / 0.37 % | +9.96 % | SAME |
| 3B 4w | 629.19 | 659.58 | +4.83 % | 0.15 / 0.10 % | +4.83 % | SAME |
| 3B 8da4w | 570.32 | 595.35 | +4.39 % | 0.14 / 0.06 % | +4.36 % | SAME |
| 8B 4w | 295.44 | 305.49 | +3.40 % | 0.06 / 0.10 % | +3.29 % | SAME |
| 8B 8da4w | 269.05 | 277.24 | +3.05 % | 0.09 / 0.04 % | +3.02 % | SAME |

Geomean **+6.13 %**, every cell outside the +-2 % band. Inside the task's expected +5 to +12 %, at its lower end.

- 60 timed runs, all valid: rc 0, 2048 prompt tokens, 0 generated, no foreign GPU process, 11 to 72 clock
  samples per window, median clock 612 MHz in every run (floor 593), **throttle state 0 in all 8051 clock samples**,
  start temperature 55 to 60 C, maximum 63 C.
- SDPA correctness, 12 passes of `all`, `extended`, `full` and 3 of `peaked`, `fused` (42 logs, 222 cases): 222
  PASSED with 0 mismatches, 222 kernel lines `qk=? softmax=? av=? fused=...fused3sb... pairing=ok` (counted from
  the logs); `gate_check.py sdpa`: ACCEPT, 0 findings.
- Unmodified `verify.sh` with the candidate environment on the timed binaries (`llama_main` `d33b5330...`,
  `test_llama_microbench` `72535d0f...`, the hashes of build `topic4`): `verify.out` (34 lines) equals the parent
  snapshot `s0-parent-verify` line by line with the rates removed (0 differing lines); 22 of 22 runner calls rc 0;
  `gate_check.py verify`: ACCEPT, 0 findings; decode 31 tokens at 18.9 / 10.9 tok/s.
- `gate_check.py session`: ACCEPT, 0 findings; next token parent vs candidate SAME in 24 of 24 rows; `env-check`:
  ACCEPT. Shipped SPIR-V of `topic4`: UNCHANGED (53 of 53).
- **How it is recorded: `ACCEPTED (reference-error rule, owner decision 2026-10-04)`**, not as a plain pass: the
  candidate replaces the three attention kernels, an arithmetic change (D3). Evidence: criterion 1 on `topic4`,
  `results/orin/sdpa-error2/` (candidate not larger than the parent in rms and maximum on all five S = 2048 cases,
  below); the real-text probe of the final stack on `topic4`, `results/orin/probe/final-fused/` (41 prompts x 6
  cells; gross-divergence check not met in any cell; logits bit-identical to `topic1`'s, `probe/c1-fused/`).
  Next-token items that differ: none (`differing-items.txt` empty). The gate's own files are kept as it wrote
  them: `gate.done` and `verify.out` of `s5-c1` read `GATE_ACCEPTED ... all steps passed`, because the gate
  script knows no other wording; that is the historical output, this line is the record.
- Memory: at least 5752 MB available before every timed run. Swap-out during 1B and 3B runs: 0 or 1 page (once
  24, parent arm). During 8B runs both arms swap out alike: parent 1 to 2872 pages per run (4 KiB each: up to
  11 MB), candidate 25 to 1268, the first run of a cell most (the model file enters the page cache, D5). The K / V
  copies do not add to it. Model load 1.6 to 10.4 s; 0 runner aborts.

Where the gain comes from (warm ETDump of both arms of `s5-c1`, ms per 2048-token prefill, parent -> candidate
1; `results/orin/sessions/s5-c1/trace/attention.csv`):

| cell | QK^T | softmax | attn*V | fused kernel | K / V copy | attention total | linear GEMM | dispatch total |
|---|---|---|---|---:|---:|---|---|---|
| 1B 4w | 107.6 -> 0 | 133.1 -> 0 | 61.4 -> 0 | 172.2 | 3.1 | 302.1 -> 175.3 | 655.3 -> 654.5 | 1359.2 -> 1225.2 |
| 1B 8da4w | 107.6 -> 0 | 133.0 -> 0 | 61.4 -> 0 | 171.9 | 3.1 | 302.1 -> 175.0 | 777.9 -> 774.4 | 1469.9 -> 1333.9 |
| 3B 4w | 198.4 -> 0 | 173.9 -> 0 | 143.6 -> 0 | 371.0 | 11.5 | 516.0 -> 382.5 | 1862.8 -> 1860.4 | 3236.6 -> 3086.8 |
| 3B 8da4w | 198.8 -> 0 | 174.2 -> 0 | 143.6 -> 0 | 371.2 | 11.3 | 516.6 -> 382.4 | 2170.8 -> 2164.8 | 3571.0 -> 3420.0 |
| 8B 4w | 301.2 -> 0 | 265.9 -> 0 | 216.1 -> 0 | 561.8 | 13.0 | 783.2 -> 574.7 | 4661.6 -> 4651.6 | 6916.6 -> 6686.1 |
| 8B 8da4w | 301.0 -> 0 | 266.0 -> 0 | 216.3 -> 0 | 561.1 | 12.9 | 783.3 -> 573.9 | 5304.2 -> 5297.0 | 7588.6 -> 7363.1 |

The same as on `topic1` within 0.5 ms: the kernel's new check costs nothing measurable. Attention falls by 42 %
on 1B, 26 % on 3B and 27 % on 8B and is 89 to 95 % of the reduction of the dispatch total. The remainder, from
the raw sums (parent -> candidate): linear GEMM -0.1 to -0.5 %, copy / view / other -0.2 to -0.7 %, everything
else (elementwise, RMSNorm, RoPE, quantize) -0.8 to -4.6 % = 4 to 13 ms per prefill (135.4 -> 129.9 ms on 1B 4w,
235.6 -> 231.2 on 1B 8da4w, 277.3 -> 264.6 on 3B 4w, 532.4 -> 522.5, 486.7 -> 476.7, 926.9 -> 919.8). Which kernels of
that remainder got faster is not located.

## On `topic4`: A/A re-check `s4-aa2` (05:05 to 05:41 UTC) and reference error `sdpa-error2`

`s4-aa2`: parent build against `topic4`, both with the parent environment, under the committed clock floor
(593 MHz), every clock sample with the throttle record. Recomputed from `runs.csv`:

| cell | parent | `topic4`, parent environment | ratio | repeat spread (parent / `topic4`) |
|---|---:|---:|---:|---|
| 1B 4w | 1490.54 | 1490.54 | 1.0000 | 0.15 / 0.15 % |
| 1B 8da4w | 1380.98 | 1385.66 | 1.0034 | 0.41 / 0.47 % |
| 3B 4w | 629.38 | 629.19 | 0.9997 | 0.12 / 0.09 % |
| 3B 8da4w | 570.47 | 570.47 | 1.0000 | 0.14 / 0.20 % |
| 8B 4w | 295.23 | 295.57 | 1.0012 | 0.20 / 0.25 % |
| 8B 8da4w | 269.08 | 269.05 | 0.9999 | 0.13 / 0.08 % |

Geomean +0.07 %, largest cell +0.34 %. 60 of 60 timed runs valid: rc 0, 2048 prompt tokens, 0 generated, no
foreign GPU process, 13 to 72 clock samples per window, median clock 612 MHz in every run, **throttle state 0 in
all 8214 clock samples of the 60 runs** (12 cooling devices), start temperature 55 to 60 C, maximum 64 C. Next
token SAME in 24 of 24 rows; `gate_check.py session`: ACCEPT, 0 findings. Parent within 0.2 % of `s10-final` in
every cell. The hook, the dev code and the kernel's new check cost nothing while the fused node is not selected.

`sdpa-error2` (one binary, `topic4`'s; same seeded inputs; stock / parent / candidate 1 / candidate 2), the five
S = 2048 cases criterion 1 judges:

| case | rms | maximum | candidate 1 / 2 not larger than the parent |
|---|---|---|---|
| `1b_head_config_s2048` | 8.547e-05 / 2.101e-05 / 2.049e-05 / 2.050e-05 | 1.713e-03 / 9.135e-04 / 7.227e-04 / 7.227e-04 | yes / yes |
| `3b_head_config_s2048` | 8.686e-05 / 2.068e-05 / 2.053e-05 / 2.053e-05 | 1.408e-03 / 7.828e-04 / 7.061e-04 / 7.061e-04 | yes / yes |
| `8b_head_config_s2048` | 8.696e-05 / 2.057e-05 / 2.022e-05 / 2.022e-05 | 1.587e-03 / 8.911e-04 / 7.911e-04 / 7.911e-04 | yes / yes |
| `tiny_gqa_s2048` | 8.357e-05 / 2.080e-05 / 2.050e-05 / 2.047e-05 | 1.498e-03 / 6.943e-04 / 6.078e-04 / 6.078e-04 | yes / yes |
| `tiny_d128_s2048` | 8.384e-05 / 2.071e-05 / 2.034e-05 / 2.034e-05 | 1.467e-03 / 7.796e-04 / 5.595e-04 / 5.595e-04 | yes / yes |

Criterion 1 is met by both candidates; 12 of 12 cases per arm with 0 mismatches. Candidate 2 differs from
candidate 1 only where head_dim is 64 (two-pass kernel on 5 of 12 cases, the same one-pass kernel on the other
7). Outside the criterion, on record, as on `topic1`: the maximum is larger than the parent's for both candidates
on `3b_head_config_s256` (8.203e-04 against 7.787e-04) and `8b_head_config_s1024_pos1024` (7.226e-05 against
6.942e-05), and candidate 1's rms on `tiny_gqa_pos64` (3.238e-05 against 3.232e-05; candidate 2: 3.220e-05).

## Build with the one-full-subgroup check: pre-check `g-pre` (04:59 to 05:04 UTC, `topic4`, `orin-fused1`)

One pass per tier: `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4, `peaked` 5 of 5, `fused` 5 of 5 PASSED; all 26
cases served by the fused kernel (`fused=...fused3sb...`, `pairing=ok` on every line). So the check does not
fire on this driver: a workgroup of 32 is one subgroup of 32 here, now tested by the kernel and no longer
assumed. The error against the fp32 reference on the five S = 2048 cases equals `topic1`'s in every printed
digit (1B rms 2.04856e-05, maximum 7.22706e-04). `chain8` wrote `topic4` into `BUILD.txt`.

## Candidate 1 under the reference-error rule: real-text evidence (`chain4`, builds `parent` and `topic1`)

`results/orin/probe/c1-fused/`: `ref_error_rule.py` verdict **MET**; no next-token item differs
(`differing-items.txt` is empty), so nothing has to be accepted under the rule; the evidence is on record as D3
asks. 41 prompts per cell, last-position logits, `compare.csv`:

| cell | top-1 differs: floor / candidate | mean KL: floor / candidate | maximum KL: floor / candidate | \|ln ppl ratio\|: floor / candidate |
|---|---|---|---|---|
| 1B 4w | 0 / 0 | 5.4e-04 / 9.0e-04 | 0.0137 / 0.0144 | 0.0126 / 0.0150 |
| 1B 8da4w | 1 / 2 | 0.0729 / 0.0410 | 0.808 / 0.291 | 0.0081 / 0.0259 |
| 3B 4w | 0 / 0 | 1.6e-04 / 5.0e-04 | 0.0026 / 0.0131 | 0.0005 / 0.0039 |
| 3B 8da4w | 1 / 0 | 0.0158 / 0.0190 | 0.103 / 0.292 | 0.0441 / 0.0005 |
| 8B 4w | 0 / 0 | 1.5e-04 / 5.0e-04 | 0.0025 / 0.0152 | 0.0021 / 0.0028 |
| 8B 8da4w | 1 / 2 | 0.0219 / 0.0170 | 0.366 / 0.118 | 0.0135 / 0.0106 |

floor = parent tiled against parent default (two arms that share the attention kernels); candidate = candidate
default against parent default.

- Gross-divergence check (D3 item 3: mean KL above 0.5 nat or top-1 differing on more than a third of the
  prompts): not met in any cell; the largest mean KL is 0.041 nat (1B 8da4w), top-1 differs on at most 2 of 41.
- Said plainly: against the older near-tie measure (D1 item 4, twice the floor) the candidate is outside in four
  cells (3B 4w and 8B 4w on KL, where the floor is 1.5e-04 nat because both parent arms run the same attention
  kernels; 3B 8da4w on maximum KL; 1B 8da4w on the perplexity ratio). D3 replaced that measure for arithmetic
  changes for exactly this reason; the numbers are in `compare.csv`.
- Criterion 1 on the cases that are not S = 2048 (outside the criterion, on record): the candidate's maximum error
  is larger than the parent's on `3b_head_config_s256` (8.203e-04 against 7.787e-04) and
  `8b_head_config_s1024_pos1024` (7.226e-05 against 6.942e-05), its rms on `tiny_gqa_pos64` (3.2378e-05 against
  3.2319e-05); smaller or equal on the other 9 of 12 in both measures.
- Peaked tier (sharp rows, the rescale path; `raw/sdpa-error1-peaked/`, recorded only): rms 3.62e-04 to 3.76e-04
  for candidate 1 and 3.65e-04 to 3.78e-04 for the parent on the five cases; maximum 2.44e-03 to 2.89e-03 against
  2.26e-03 to 3.02e-03.
- This evidence is for `topic1`'s kernel. The probe was repeated on `topic4` (`probe/final-fused/`, above): the
  logits are bit-identical.

## Final stack: the rule, fixed 01:22 UTC before `s5-c1` and `s6-c2` have a number

`chain7.sh` applies it without me: the final stack is `orin-fused2` only if `s6-c2` is `GATE_ACCEPTED` and its
geomean gain over candidate 1 is at least 2 % (outside the noise band); in every other case it is `orin-fused1`,
provided `s5-c1` is accepted (otherwise the chain stops and nothing is final). This is the owner's sentence of
01:00 UTC ("if `s6-c2` is inside the band, candidate 1 alone is the final stack") read on the geomean, the
quantity R11 stops on. If the two 1B cells alone come out above 2 % while the geomean stays under it, candidate 2
is still not adopted and the cells are reported as measured. Either way the campaign stops after candidate 2
(`thresholds.txt`, "stop").

The build that is measured as final is the one `chain8` chooses (`topic4` = `0bed38090`, the last commit that
changes code; `topic3` = `ca62778e6` only if the check below fails its pre-check). Later commits change only this
change directory (tools, evidence, text): `git diff <that commit> HEAD` outside
`openspec/changes/sarc-1.5-orin-fused-port/` is empty and is checked again at closing.

## Decision needed from the owner

**Candidate 2 goes beyond the clause I fixed for it.** `tools/thresholds.txt` (committed before any measurement)
says candidate 2 exists only if the unpacked one-pass form is at least 3 % faster at kernel level, and otherwise
there is none. That clause is answered: the unpacked forms are 1.5 to 3.4 times slower (`sdpa-screen1`), so by
it there is no candidate 2. The same screen, which measured all four forms of the ported kernel, showed
something the clause did not anticipate: for head_dim 64 the **two-pass** packed form takes 17 % less time than the
one-pass form in every round (9.62 against 11.59 ms per 1B layer); for head_dim 128 the one-pass form stays
faster (13.9 against 19.2 ms, 18.3 against 25.3). The task's own candidate-2 list names the one-pass / two-pass
choice (in the other direction), R8 says "choose the best kernel per shape" with the 3 % in every round margin,
and no new kernel, tile or search is involved: both forms are variants of the one ported shader, already built.
So I gate it as candidate 2 (`orin-fused2` = two passes for head_dim 64, one pass for 128), expecting about
+2 % on the two 1B cells and nothing elsewhere (under +1 % geomean), and I say here that this is my reading, not
the letter of my own pre-registered clause, which I have not edited.

**Answered by the owner, 2026-10-09 01:00 UTC (task file, "candidate 2 as you read it"):** agreed; gate
`orin-fused2` as candidate 2; the pre-registered clause stays unedited and this note stays. If `s6-c2` is inside
the band, candidate 1 alone is the final stack and the campaign closes by N3 with candidate 2 as the first
sub-threshold candidate. Nothing is open under this heading now.


## Earlier: candidate 1 on `topic1`, gate `s3-c1`, `GATE_ACCEPTED 2026-10-09T00:34:54Z all steps passed` (superseded by `s5-c1`; kept as measured)

Parent arm: build `parent` (`8973ced76`) with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5
ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`. Candidate arm: build `topic1` (`0f14f2a1a`) with `ET_VK_SARC_UNVERIFIED=1
ET_VK_SARC_DEV_PROFILE=orin-fused1`. Tok/s, median of 5 valid interleaved runs per arm; every number below
recomputed from `runs.csv` (`results/orin/sessions/s3-c1/`):

| cell | parent | candidate 1 | gain | repeat spread (parent / candidate) | next token (4 prompts) |
|---|---:|---:|---:|---|---|
| 1B 4w | 1489.45 | 1651.61 | +10.89 % | 0.07 / 0.08 % | SAME |
| 1B 8da4w | 1383.78 | 1521.55 | +9.96 % | 0.47 / 0.22 % | SAME |
| 3B 4w | 628.99 | 659.37 | +4.83 % | 0.22 / 0.23 % | SAME |
| 3B 8da4w | 570.16 | 595.00 | +4.36 % | 0.06 / 0.06 % | SAME |
| 8B 4w | 295.40 | 305.12 | +3.29 % | 0.10 / 0.10 % | SAME |
| 8B 8da4w | 269.08 | 277.21 | +3.02 % | 0.05 / 0.08 % | SAME |

Geomean **+6.01 %**, every cell outside the +-2 % band. Inside the task's expected +5 to +12 %, at its lower end.

- 60 timed runs, all valid by the rules of that session: rc 0, 2048 prompt tokens, 0 generated, no foreign GPU
  process, 12 to 73 clock samples per window, median clock 612 MHz in every run (floor 593), start temperature
  57 to 62 C. **Limitation: no thermal-throttle record exists for these runs** (below); `s5-c1` repeats the
  gate with it.
- SDPA correctness, 12 passes of `all`, `extended`, `full` and 3 of `peaked`, `fused` (42 logs, 222 cases): 0
  mismatches, every case served by the fused kernel alone (`qk=? softmax=? av=?`, `pairing=ok`);
  `gate_check.py sdpa`: ACCEPT, 0 findings.
- Unmodified `verify.sh` with the candidate environment on the timed binaries: `gate_check.py verify` against
  `s0-parent-verify` ACCEPT, 0 findings; `verify.out` equals the parent snapshot line by line with the rates
  removed (0 differing lines); all four default-vs-tiled items SAME; decode 31 tokens (18.9 / 10.8 tok/s, the
  parent's 19.1 / 10.9: decode is not served by the fused node).
- `gate_check.py session`: ACCEPT, 0 findings; next token parent vs candidate SAME in 24 of 24 rows;
  `env-check`: ACCEPT. Shipped SPIR-V of `topic1`: UNCHANGED (53 of 53).
- **How it is recorded:** the gate wrote `all steps passed` because no next-token item differs. The candidate
  replaces the three attention kernels, an arithmetic change, so it is recorded under the reference-error rule
  and not as a plain pass, once `chain4` has produced the real-text evidence. Criterion 1 as fixed in
  `thresholds.txt` (candidate not larger than this campaign's parent in rms and in maximum on all five S = 2048
  cases) is met (`results/orin/sdpa-error1/`, one binary, same inputs):

  | case (S = 2048) | rms: stock / parent / candidate 1 | maximum: stock / parent / candidate 1 |
  |---|---|---|
  | `1b_head_config_s2048` | 8.547e-05 / 2.101e-05 / 2.049e-05 | 1.713e-03 / 9.135e-04 / 7.227e-04 |
  | `3b_head_config_s2048` | 8.686e-05 / 2.068e-05 / 2.053e-05 | 1.408e-03 / 7.828e-04 / 7.061e-04 |
  | `8b_head_config_s2048` | 8.696e-05 / 2.057e-05 / 2.022e-05 | 1.587e-03 / 8.911e-04 / 7.911e-04 |
  | `tiny_gqa_s2048` | 8.357e-05 / 2.080e-05 / 2.050e-05 | 1.498e-03 / 6.943e-04 / 6.078e-04 |
  | `tiny_d128_s2048` | 8.384e-05 / 2.071e-05 / 2.034e-05 | 1.467e-03 / 7.796e-04 / 5.595e-04 |

  Outside the criterion's cases, on record: `8b_head_config_s1024_pos1024` maximum 6.942e-05 -> 7.226e-05
  (candidate 4 % larger), rms 9.018e-06 -> 8.980e-06. The 4070 Ti port measured the same numbers to three digits
  with the same kernels (its STATUS: 1B 2.101e-5 / 2.049e-5, 9.14e-4 / 7.23e-4): the two NVIDIA drivers agree.

Where the gain comes from (warm ETDump of both arms of `s3-c1`, ms per 2048-token prefill, parent -> candidate
1; `results/orin/sessions/s3-c1/trace/attention.csv`, `tools/trace_kernels.py`):

| cell | QK^T | softmax | attn*V | fused kernel | K / V copy | attention total | linear GEMM | dispatch total |
|---|---|---|---|---:|---:|---|---|---|
| 1B 4w | 107.6 -> 0 | 133.1 -> 0 | 61.5 -> 0 | 172.4 | 3.1 | 302.2 -> 175.5 | 655.5 -> 654.6 | 1360.0 -> 1225.1 |
| 1B 8da4w | 107.6 -> 0 | 133.0 -> 0 | 61.5 -> 0 | 172.2 | 3.1 | 302.1 -> 175.3 | 778.1 -> 774.3 | 1468.5 -> 1334.5 |
| 3B 4w | 198.5 -> 0 | 174.1 -> 0 | 143.8 -> 0 | 371.4 | 11.5 | 516.4 -> 382.9 | 1862.4 -> 1861.0 | 3236.9 -> 3088.4 |
| 3B 8da4w | 198.9 -> 0 | 174.3 -> 0 | 143.7 -> 0 | 371.1 | 11.2 | 516.9 -> 382.4 | 2171.0 -> 2165.4 | 3571.3 -> 3421.4 |
| 8B 4w | 301.3 -> 0 | 266.0 -> 0 | 216.3 -> 0 | 561.6 | 13.0 | 783.5 -> 574.6 | 4657.4 -> 4656.6 | 6914.1 -> 6690.8 |
| 8B 8da4w | 301.2 -> 0 | 266.0 -> 0 | 216.1 -> 0 | 561.0 | 12.9 | 783.3 -> 573.9 | 5305.1 -> 5302.1 | 7591.0 -> 7368.3 |

Nearly all of the gain is attention (89 to 95 % of the reduction; see the `s5-c1` section for the remainder): -42 % on 1B, -26 % on 3B, -27 % on 8B (the RX 7600 saw -76 %, the 4070 Ti -66 %
and -47 to -49 %). The copy pass is small: 3 to 13 ms per prefill, 0.19 ms a layer on 1B and 0.41 ms on 3B / 8B.
Per layer the fused kernel takes 10.8 / 13.3 / 17.5 ms. The same 17.2 GFLOP of a 1B layer would take 1.8 ms at
this device's fp16 -> fp32 matrix roof (9.7 TFLOP/s): the kernel runs at 16 % of it, where the 780M ran at 71 %
and the 4070 Ti at 53 to 63 %. The first campaign measured why: this device feeds its matrix unit from DRAM at
2.8 TFLOP/s against 8.8 from shared memory, and this kernel loads K and V tiles straight from DRAM (its design:
no shared staging). That is what limits it here (and what a staged structure would address; not in this
campaign's scope).

## Kernel level: the four forms of the ported kernel (`sdpa-screen1`, 22:12 to 22:25 UTC, build `topic1`)

`test_llama_microbench --sdpa`, ms per layer at S = 2048 (copy pass + fused kernel; the parent: QK^T + softmax +
attn*V), 3 rounds interleaved, every round listed (`results/orin/screens/sdpa-screen1.csv`):

| form | 1B (head_dim 64, `t32x32`) | 3B (head_dim 128, `t16x64`) | 8B (head_dim 128, `t16x64`) |
|---|---|---|---|
| parent, three kernels | 18.87 / 18.89 / 18.90 | 18.45 / 18.46 / 18.45 | 24.45 / 24.48 / 24.48 |
| `rko` one pass, packed (**candidate 1**) | 11.62 / 11.53 / 11.59 | 13.84 / 13.90 / 13.90 | 18.25 / 18.27 / 18.25 |
| `rk` two passes, packed | **9.61 / 9.62 / 9.64** | 19.19 / 19.20 / 19.26 | 25.26 / 25.27 / 25.33 |
| `ro` one pass, unpacked | 17.21 / 17.26 / 17.29 | 47.11 / 47.08 / 47.19 | 62.27 / 62.28 / 62.47 |
| `r` two passes, unpacked | 23.56 / 23.43 / 23.47 | 66.46 / 66.51 / 66.35 | 88.28 / 88.35 / 88.42 |

- The copy passes save far more than they cost: unpacked is 1.5 times (head_dim 64) and 3.4 times (128) slower.
  By the clause of `thresholds.txt` there is no "unpacked" candidate.
- One pass against two: for head_dim 128 the one-pass form takes 28 % less time, as on the 780M; for head_dim 64 it is
  20.4 % slower (median 11.59 against 9.62 ms; the two-pass form takes 17.0 % less time and clears the 3 % margin
  in every round). On the 780M the one-pass form won for both. The two-pass form computes the scores twice, but it
  declares 2 KB less shared memory per workgroup (no rescale divisors) and has no rescale branch; the first
  campaign measured that a workgroup's time on this device grows with the shared memory it declares. That is an
  observation that fits, not a measured cause.

## What I took from the 4070 Ti fused port (`origin/topic/4070ti-fused-port`, read 00:38 UTC at `ed8b5af91`)

Its campaign closed at +11.64 % and was reopened by its review for three things. Checked against this campaign:

| its finding | here | what I did |
|---|---|---|
| the gate ran no 12 passes of tier `all` | not affected: `gate_sdpa.sh` here runs 12 of `all`, `extended` and `full` since the first gate | nothing |
| no timed run recorded a thermal-throttle reason, so R6's "no thermal throttle reason" was never evaluated | **affected**: the sampler recorded clock, load, power and the GPU temperature, no throttle state. `s1-aa` and `s3-c1` are kept as measured WITH THAT LIMITATION (GPU temperature at most 66 C against trip points of 70 C (alert) and 99 C (throttle), and a 612 MHz clock in every run; that is context, not a record) | every clock sample now carries the state of the 12 thermal cooling devices of the module (`cpufreq-cpu0/4`, `devfreq-17000000.gpu`, the `*-throttle-alert` devices, `hot-surface-alert`; the fan is left out); a timed run with a nonzero state or without the record is invalid (`runrow.py`, 6 new unit tests, 54 in all). A/A re-check `s4-aa2` and the gate again (`s5-c1`) under it; thresholds unchanged |
| the shared test `test_llama_microbench.cpp` was edited in place (R3) | **affected**: my first form changed 9 existing lines | rewritten as seven insert-only blocks delimited by `// >>> orin-fused <id>` (`ca62778e6`; `git diff 8973ced76 -- backends`: 0 deleted lines in all 13 files); same cases, same seeded inputs. New build `topic3`; everything reported as final is gated on it |

| (not a review finding; its kernel has it, mine did not) its `fused3sb` returns unless `gl_NumSubgroups == 1` and `gl_SubgroupSize == SUBGROUP_SIZE` | **affected**: this port's kernel was the RX 7600's text unchanged and only assumed that a workgroup of 32 is one full subgroup. The pipeline asks for subgroup size 32 (`REQUIRED_SUBGROUP_SIZE`, `vk_api/Pipeline.cpp`) but does not set `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`, which is what the specification names for it ("specifies that the subgroup sizes must be launched with all invocations active in the task, mesh, or compute stage", refpage `VkPipelineShaderStageCreateFlagBits`, `vulkan-docs`); that code is release zone. With two subgroups in a workgroup `Psh`, `Rsh` and `Dsh` would be shared between them and `subgroupBarrier()` would order nothing across them. 222 of 222 correct cases on `topic1` say it does not happen on this driver; nothing guarantees it | the same five lines (commit `0bed38090`, read 01:24 UTC): the kernel writes nothing unless the workgroup is one subgroup of 32, so a driver that splits it fails the correctness tiers instead of racing. Both conditions depend on the workgroup only (`gl_NumSubgroups` "is guaranteed to be uniform across a shader execution", GL_KHR_shader_subgroup), so the return is uniform and precedes every barrier. New build `topic4`; `chain8` checks it first (`g-pre`) |

Also taken: its numbers for comparison. Same kernel, same architecture family: +11.64 % there (1B +19 / +21 %,
3B +8 / +10 %, 8B +6 / +7 %) against +6.0 % here; its fused kernel removes 66 % / 47 to 49 % of attention time,
here 42 % / 26 to 27 %; its copy pass 0.1 to 0.4 ms per prefill, here 3 to 13 ms. Its reference errors equal
mine to three digits.

## Specification text the shader reading rests on (owner note 2026-10-09; R7)

From the `vulkan-docs` index (built 2026-10-08T23:04:58Z from docs.vulkan.org). The MCP server was not attached
to this session, so I called it over stdio from the shell (`search_docs`, `get_page`, `spirv_opcode`); the pages
are `spec/latest/memorymodel.md` and `glslext/latest/GL_KHR_shader_subgroup.md`.

- What must not happen (memory model, "Data Race"): "Let X and Y be operations that access overlapping sets of
  memory locations M, where X != Y, and at least one of X and Y is a write, and X and Y are not mutually-ordered
  atomic operations. If there does not exist a location-ordered relation between X and Y for each location in M,
  then there is a data race. Applications must ensure that no data races occur during the execution of their
  application."
- What `subgroupBarrier()` is (GL_KHR_shader_subgroup): "subgroupBarrier() -> OpControlBarrier( /*Execution*/
  Subgroup, /*Memory*/Subgroup, /*Semantics*/AcquireRelease | UniformMemory | WorkgroupMemory | ImageMemory)",
  and "For each active invocation within a subgroup that reaches the same dynamic instance of a subgroup
  built-in function, all active invocations within a subgroup must execute the dynamic instance of the function
  before any invocation can proceed."
- Why a store before it is ordered before another lane's load after it (memory model): "If A is a release
  barrier, B is an acquire barrier, and C is a control barrier (where A can equal C, and B can equal C), then A
  synchronizes-with B if all of the following are true: A is program-ordered before (or equals) C; C is
  program-ordered before (or equals) B; A and B are in the instance of each other's memory scopes; A and B are
  in the instance of C's execution scope."

In the kernel every exchange through `Psh`, `Rsh` and `Dsh` has such a barrier between the store and the other
lane's load (in the SPIR-V: `OpControlBarrier %uint_3 %uint_3 %uint_3400`, execution and memory scope Subgroup,
semantics AcquireRelease | UniformMemory | WorkgroupMemory | ImageMemory, one after each `OpMemoryBarrier`; 7
pairs in `d64 rko`, 9 in `d128 rko` and in `d64 rk`), and no location has two writers between two barriers (the
table under "Shared-memory reading" below). `memoryBarrierShared()` alone, which the 780M's `fused3` uses, is a
memory barrier and no control barrier: it does not make the other lanes arrive, which is why this port starts
from `fused3sb`. A workgroup is one subgroup here, so the subgroup barrier is the only one needed.

## Hook control `s2n-noenv` (21:35 to 21:52 UTC): `GATE_ACCEPTED`, owner decision D4

Unmodified `verify.sh` on build `topic1` (`0f14f2a1a`: the hook `9d91480b2` + the whole dev zone of this
campaign) with **nothing selected** (no environment), against `s0n-noenv` (the parent build, nothing selected):

- `verify.out` equal line by line with the rates removed: **0 differing lines** (`verify-lines.txt`);
- `gate_check.py verify` against `s0n-noenv`: ACCEPT, 0 findings (every correctness case, production-diff shape
  and linear dispatch state equal); 22 of 22 runner calls rc 0; no `[sarc_dev]` banner in any log;
- default-arm prefill runs 890.44 / 821.83, 360.37 / 320.15, 189.67 / 170.50 tok/s (`s0n-noenv`: 890.82 / 823.15,
  360.25 / 320.20, 189.63 / 170.33);
- `test_sarc_select` on the release tables: `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent,
  unverified off)`, the parent's line; shipped SPIR-V: all 53 variants byte-identical to the parent build;
- with the parent environment selected instead, `topic1` times like the parent build: the A/A below.

So with nothing selected the branch dispatches what the parent dispatches. The entry point stays subject to the
owner's review before any promotion (hook D4.3).

## Candidate 1 pre-check (`raw/c1-pre/`, 21:54 to 22:05 UTC, one pass per tier, build `topic1`)

`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`. Banners: `softmax variant: orin_g64`, `profile active:
orin-fused1`, `orin fused attention: ...d64_t32x32g11s32rko ...d128_t16x64g11s32rko`.

| tier | cases | PASSED, 0 mismatches | kernels |
|---|---:|---:|---|
| `all` | 4 | 4 | fused only (`qk=? softmax=? av=?`, `pairing=ok`) |
| `extended` | 8 | 8 | fused only |
| `full` | 4 | 4 | fused only |
| `peaked` (sharp rows: the rescale path) | 5 | 5 | fused only |
| `fused` (S = 32, 64, 192, 320: shapes no three-kernel tile fits) | 5 | 5 | fused only |

Error against the fp32 CPU reference on the S = 2048 cases, parent (from `s0-parent-verify`) -> candidate
(this pass); the formal comparison on one binary is `sdpa-error1`:

| case | rms parent -> candidate | maximum parent -> candidate |
|---|---|---|
| `1b_head_config_s2048` | 2.101e-05 -> 2.049e-05 | 9.135e-04 -> 7.227e-04 |
| `3b_head_config_s2048` | 2.068e-05 -> 2.053e-05 | 7.828e-04 -> 7.061e-04 |
| `8b_head_config_s2048` | 2.057e-05 -> 2.022e-05 | 8.911e-04 -> 7.911e-04 |
| `tiny_gqa_s2048` | 2.080e-05 -> 2.050e-05 | 6.943e-04 -> 6.078e-04 |
| `tiny_d128_s2048` | 2.071e-05 -> 2.034e-05 | 7.796e-04 -> 5.595e-04 |

Not an S = 2048 case and so outside the criterion as fixed in `thresholds.txt`, but on record:
`8b_head_config_s1024_pos1024` reads rms 9.018e-06 -> 8.980e-06 and maximum 6.942e-05 -> 7.226e-05 (the
candidate's maximum is 4 % larger there).

## Baseline and A/A (`s1-aa`, 20:35 to 21:20 UTC, record-only clock)

Parent arm = build `parent` (`8973ced76`), candidate arm = build `topic1` (`0f14f2a1a`: hook + candidate 1
code), **both with the parent environment** (`orin-refine5` + `orin_g64`): the A/A also shows that the hook and
the linked dev code cost nothing while the fused node is not selected. Tok/s, median of 5 valid runs per arm,
arms interleaved; recomputed from `runs.csv`:

| cell | parent | `topic1`, parent environment | ratio | `s10-final` (expected) | parent vs expected | repeat spread (parent / `topic1`) |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 1489.45 | 1488.37 | 0.9993 | 1489.45 | 0.00 % | 0.07 / 0.15 % |
| 1B 8da4w | 1382.85 | 1380.05 | 0.9980 | 1382.85 | 0.00 % | 0.34 / 0.47 % |
| 3B 4w | 629.38 | 629.19 | 0.9997 | 628.99 | +0.06 % | 0.15 / 0.12 % |
| 3B 8da4w | 570.16 | 570.16 | 1.0000 | 570.32 | -0.03 % | 0.14 / 0.08 % |
| 8B 4w | 295.27 | 295.27 | 1.0000 | 295.53 | -0.09 % | 0.19 / 0.12 % |
| 8B 8da4w | 269.05 | 269.12 | 1.0003 | 269.19 | -0.05 % | 0.08 / 0.09 % |

- Baseline: every cell within 0.09 % of the first campaign's final session (threshold 3 %).
- A/A: geomean -0.05 %, largest cell -0.20 %, repeat spread at most 0.47 %. The noise on this device is far
  inside the +-2 % band. The runner's timer has a 1 ms step: 0.07 % of a 1B prefill (1375 ms).
- 60 timed runs, all valid: rc 0, 2048 prompt tokens, 0 generated, no foreign GPU process, 13 to 73 clock
  samples per prefill window (threshold 5), median clock 612 MHz in every run. Start temperature 57 to 62 C.
  Next token parent vs `topic1`: SAME in 24 of 24 rows. `gate_check.py session --calibration --require-logs`:
  ACCEPT, 0 findings.
- Clock floor (`results/orin/clkmin.json`, `calibrate_clock.py`): floor(0.97 x 612) = **593 MHz**, device-wide; no
  run below it.
- Memory: at least 5628 MB available before every timed run; swap in use (74 and 106 MB at two looks) since the first 8B run of
  the parent control (the model file is read into the page cache before each cell, D5). Model load 1.6 to 10.8 s,
  none slow enough to abort: 0 runner aborts in the session.

## Pristine control `s0n-noenv` (20:35 UTC, `GATE_ACCEPTED`)

Unmodified `verify.sh` on the `parent` build with nothing selected: the dev/1.5 state of this device.
`gate_check.py verify`: ACCEPT, 0 findings. Its default-arm prefill runs: 1B 890.82 / 823.15, 3B 360.25 / 320.20,
8B 189.63 / 170.33 tok/s; the published `cells.csv` numbers are 890.82 / 822.82, 360.37 / 320.30, 189.74 / 170.43:
within 0.1 %. Error of the stock attention kernels against the fp32 reference on the S = 2048 cases: rms 8.4e-05
to 8.7e-05, maximum 1.41e-03 to 1.71e-03 (four times and twice the parent's).

## Builds

Cross image `localhost/et-jetson-cross:jp7.2.1` (`d182f725bb32`), shaderc v2026.1, 8 jobs, exclusive build lock;
provenance `.artifacts/build/<tag>.src.txt`.

| tag | commit | source | SPIR-V |
|---|---|---|---|
| `parent` | `8973ced76` | `git archive` + 30 pinned submodules, tree sha256 `3bbbd4cb...` | 1610 shaders, **all byte-identical to the first campaign's final build `topic14`** (`diff` of the two `spv.sha256` lists: 0 lines); `libllama_runner.so` has `topic14`'s hash |
| `topic1` | `0f14f2a1a` | hard links to `parent` + 77 changed paths, each verified by blob hash | 1620 shaders: the parent's 1610 byte-identical + the 10 new `sarc_dev_orin_sdpa_*`; `tools/shipped.py`: all 53 shipped variants byte-identical to `parent`: **UNCHANGED** |
| `topic2` | `c6be297f1` (profile `orin-fused2` added) | hard links to `topic1` + changed paths | 1620 shaders, all byte-identical to `topic1`; shipped: **UNCHANGED**. Deployed, used for nothing (superseded by `topic3` before any job ran on it) |
| `topic3` | `ca62778e6` (test support as insert-only blocks) | hard links to `topic2` + 104 changed paths | 1620 shaders; shipped: **UNCHANGED** (53 of 53 equal to `parent`; `build/topic3.shipped.txt`); `llama_main` `d37d44dd...`, `test_llama_microbench` `c1cfe15b...` |
| `topic4` | `0bed38090` (the one-full-subgroup check in the fused kernel; the last commit that changes code) | hard links to `topic3` + 7 changed paths | 1620 shaders: the 8 `fused3sb` variants differ from `topic3`, the other 1612 are byte-identical; shipped: **UNCHANGED** (53 of 53 equal to `parent`; `build/topic4.shipped.txt`); `llama_main` `d33b5330...`, `test_llama_microbench` `72535d0f...` |

| `parentg` | `8973ced76`, shaders compiled with the pinned `glslc` (`GLSLC_DIR`) | as `parent` | `spirv golden: PASS (53 shipped variants)`. **Not measured** (owner decision 15:50 UTC); only the interrupted first step of `chain10` ran on it |
| `topic4p` | `0bed38090`, the same compiler | as `topic4` | `spirv golden: PASS (53 shipped variants)`; its 1620 shaders equal the pinned image's output byte for byte. **Not measured**; nothing ran on it |
| `topic4g` | `0bed38090` | | build failed (rc 141), not used |

`spirv_golden.py` reads FAIL with 14 DIFF lines on the five measured builds (`parent`, `topic1` to `topic4`), as on every build of the first campaign: the
cross image's glslc is not the one the goldens were made with; none of the 14 is a kernel the Orin rows
dispatch (`build/topic1.shipped.txt`: same set in parent and candidate, Orin kernels differing: 0).

## Parent control `s0-parent-verify` (20:14 UTC, `GATE_ACCEPTED`)

Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` on the `parent` build with the
parent environment. `gate_check.py verify` against itself: ACCEPT, 0 findings. What it prints (and every
candidate must print again): `correctness rc=1`, `linear 4w rc=1`, `linear 8da4w rc=1`, the six `pdiff ... buffer
rc=1` lines (no buffer row for this device; the same lines as in the first campaign), six `pdiff ... texture3d`
ALL PASSED, default vs tiled SAME on the check and the unaligned prompt for 1B 4w and 8da4w, decode 31 tokens.
Its single default-arm prefill runs: 1B 1489.45 / 1379.12, 3B 628.99 / 570.47, 8B 294.85 / 269.01 tok/s
(expected 1489.45 / 1382.85, 628.99 / 570.32, 295.53 / 269.19): within 0.3 %.

Error of the parent's attention kernels against the fp32 CPU reference (one pass per tier, recorded with the
control; this is what criterion 1 of the reference-error rule compares the candidate with):

| case (S = 2048) | rms | maximum |
|---|---:|---:|
| `1b_head_config_s2048` | 2.101e-05 | 9.135e-04 |
| `3b_head_config_s2048` | 2.068e-05 | 7.828e-04 |
| `8b_head_config_s2048` | 2.057e-05 | 8.911e-04 |
| `tiny_gqa_s2048` | 2.080e-05 | 6.943e-04 |
| `tiny_d128_s2048` | 2.071e-05 | 7.796e-04 |

## Parent

`8973ced76` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
(the first campaign's final stack; the softmax hook `307abb2ed` is committed, so the softmax is named by the
dev-zone variable, no local patch). Expected (`s10-final`): 1B 1489.45 / 1382.85, 3B 628.99 / 570.32,
8B 295.53 / 269.19 tok/s (4w / 8da4w). Pristine state: the same commit, no environment.

## Ceiling (computed before candidate 1, from the first campaign's `s10-final` traces)

Attention (QK^T + softmax + attn*V) in the parent, ms per 2048-token prefill, and its share of the dispatch
total (`sarc-1.5-orin-prefill-refine/proposal.md`, "Where the gain is"):

| | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention / total, ms | 302 / 1359 | 302 / 1465 | 516 / 3235 | 517 / 3571 | 783 / 6913 | 784 / 7587 | |
| share | 22.2 % | 20.6 % | 16.0 % | 14.5 % | 11.3 % | 10.3 % | |
| gain if attention cost nothing (ceiling) | +28.6 % | +26.0 % | +19.0 % | +16.9 % | +12.8 % | +11.5 % | +19.0 % |
| gain if 76 % of it goes (the RX 7600's fused kernel) | +20.3 % | +18.6 % | +13.8 % | +12.4 % | +9.4 % | +8.5 % | +13.7 % |
| gain if 50 % of it goes | +12.5 % | +11.5 % | +8.7 % | +7.8 % | +6.0 % | +5.4 % | +8.6 % |

The task expects +5 to +12 % geomean. On this device attention is bound by memory traffic (the first campaign:
half of the softmax is the first read of a row) and shared memory is expensive, so the lower half is as likely
as the upper.

## Candidate 1: what was written (commit `0f14f2a1a`, build `topic1`)

As written before the first measurement. Changed since: the microbench support became insert-only blocks
(`ca62778e6`), and the fused kernel got the one-full-subgroup check (`0bed38090`), so it is the RX 7600's text
plus five lines and no longer identical to it.

- Hook: `9d91480b2` = cherry-pick of `1c8861aa7e`, alone, same patch-id (`0027e89f...`), 49 lines in `SDPA.cpp`,
  `sarc/SdpaCoopmat.{cpp,h}`, `sarc/Select.h`. `sarc/tools/check.sh --no-build` on `0f14f2a1a`: `check.sh: PASS`;
  `test_sarc_select` on the release tables `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent,
  unverified off)`, the parent's line.
- `glsl/sarc_dev/sarc_dev_orin_sdpa_fused3sb.{glsl,yaml}`: the RX 7600's `fused3sb` (`b3bb758e38`), identical from
  `#version` on (`diff`: no difference); `sarc_dev_orin_sdpa_{kvt,vt}`: the 780M's copy passes, identical from
  `#version` on. Variants: `rko` (one pass, packed: candidate 1), `ro` (one pass, unpacked), `rk`, `r` (two
  passes) for head_dim 64 (`t32x32`) and 128 (`t16x64`).
- `impl/sarc_dev/orin/SdpaOrinFused.cpp`: port of the 780M's selector; sets `Override::sdpa_fused_*` only when a
  variant is named (profile `orin-fused1` or `ET_VK_SARC_ORIN_SDPA_FUSED`), so with nothing selected both are null.
- `Overrides.cpp`, blocks `orin-fused profiles` / `orin-fused override`: profile `orin-fused1` = the preferences
  of `orin-refine5`; an `orin-fused*` profile names the softmax `orin_g64` itself. Candidate environment:
  `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-fused1`.
- `test_llama_microbench.cpp`: the fused kernel is recognised (timing, `fused=` in `[sdpa-kernels]`, pairing: no
  QK^T / softmax / attn*V kernel may run beside it); the 780M's tiers `peaked` and `fused` added.

### Shared-memory reading of the kernel (R7, before any gate)

Read in the source and in the SPIR-V of all eight variants (host glslc, shaderc v2026.1; the ten binaries of
build `topic1` have the same sha256): every `OpMemoryBarrier` is followed by `OpControlBarrier` with execution scope Subgroup (7 pairs in the
`t32x32` one-pass variants, 9 in `t16x64`).

| shared object | writer | readers | ordered by |
|---|---|---|---|
| `Psh` scores | `coopMatStore` of the whole subgroup (`qk_block`) | each lane, its own row segment | barrier pair after `qk_block` |
| `Psh` e | each lane, its own segment `e_idx + [0, SEG_V8)`: row `id % M`, segment `id / M`, one lane per slot | `coopMatLoad` (`av_block`) | barrier pair before `av_block`; another after it, before the next `qk_block` store |
| `Rsh` (row maximum per block, row sum) | each lane, slot `e_row * SEGS + e_seg`: one lane per slot | the lanes of the same row | barrier pair after the store, another after the reads |
| `Dsh` rescale divisors | each lane, `e_row * 4 + j`, `j = e_seg, e_seg + SEGS, ...`: disjoint per lane | `coopMatLoad` | barrier pair after the stores, another after the load; inside `if (subgroupAny(...))`, which is uniform over the subgroup |
| `Psh` divisors (end) | each lane, `e_row * P_STRIDE + j`, disjoint per lane | `coopMatLoad` | barrier pair after the stores |

No slot has two writers (no "every lane stores the reduced value"); no read of another lane's slot without an
execution barrier after its store; no write to a slot another lane may still be reading (a barrier pair closes
every read phase). Row ownership uses `gl_SubgroupInvocationID` only, never `gl_LocalInvocationID`. What the
kernel still assumes, as every cooperative-matrix kernel on this device does: a workgroup of 32 is exactly one
full subgroup. The copy passes use no shared memory; every invocation writes its own 4 x 4 block.
Not lockstep-dependent, but worth watching in the tiers: the one-pass form starts from a row maximum of -inf
(`exp(+inf)`, `0 / inf`); the `peaked` tier exercises the rescale path.

## `sarc/tools/check.sh --no-build` at closing

**Not a clean PASS at closing.** Run again 2026-10-09 16:06 UTC on the workstation, unmodified: step 1 prints
`FAIL: .agents/skills/review-notes/SKILL.md is outside the zones and not in sarc/HOOKS`, both selector tests pass
(1240 checks / 31 rows / 0 candidates; 1593 checks / 33 rows / 261 candidates), `check.sh: FAIL`, rc 1. The file
is untracked, dated 2026-10-04, placed in the working copy by the review tooling (it was absent for a few
minutes around 15:50 UTC, when the check printed PASS, and was back at 15:51 UTC); it is not part of this branch
and I have neither committed, edited nor removed it, nor touched `check.sh` or `sarc/HOOKS`. See "For the
coordinator" at the top. The output below is the run of 15:05 UTC, with the file absent, steps 4 and 5 skipped by
`--no-build` (the shipped SPIR-V of the cross build is compared by `tools/shipped.py`, above):

```
== 1 zone rule vs origin/release/1.5
== 2 twin wrappers
== 3 test_sarc_select
test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
[sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
test_sarc_select: PASS (1593 checks, 33 rows, 261 candidates, dev zone linked, unverified on)
check.sh: PASS
rc=0
```

It does not report the release-zone hook (`impl/sarc/` is inside its zones, `impl/SDPA.cpp` is in `sarc/HOOKS`): the
four files of commit `9d91480b2` are named in `proposal.md` under the owner decision of 2026-10-05.

## Next step

None in this campaign. Open to the owner (not asked for here): the review of the release-zone entry point
(hook D4.3) before any promotion; a staged fused kernel for this device (above); whether `orin-fused2`'s +1.7 % on
1B is wanted in spite of the band.

## Thresholds

`tools/thresholds.txt` (committed in `086651dd4`, before any measurement).
