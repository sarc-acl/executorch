# sarc-1.5-orin-prefill-refine

Dev zone only (2026-10-05). Branch `topic/orin-prefill-refine`, forked from `topic/4070ti-prefill-refine` at
`42e001462`. The parent of every comparison is commit `6a7cc8cc6` with no profile (the shipped NVIDIA rows).
Nothing is promoted; no release-zone or upstream file is edited. Day-to-day state, incidents and the tool
history are in `STATUS.md`.

## Why

Push the 2048-token Llama prefill on the Jetson Orin Nano (8 GB, 15 W mode, Vulkan driver NVIDIA 595.78) as far
as it goes, measured against the parent in the same session.

## Outcome

DRAFT: the candidate gates are running; this section is written when they have ended. See `STATUS.md`.

## What is in the tree

All under `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/` and this
directory. Every file of this campaign carries `orin` in its name; `impl/sarc_dev/Overrides.cpp` is changed only
inside `// >>> orin <id>` .. `// <<< orin <id>` blocks (`tools/devzone.py` enforces both). The generators read
the release sources and the sibling campaign's dev sources and never write them.

| file / family | what | generator |
|---|---|---|
| `impl/sarc_dev/OrinSdpa.cpp` | two `kUnverified` SDPA base rows for `tegra orin` that match only while `ET_VK_SARC_DEV_PROFILE` names an `orin-*` profile other than `orin-lin-*`: the SARC SDPA path without a hook (the Xe2 mechanism) | `gen_orin_sdpa.py` |
| `orin-qk-*`, `orin-av-*` profiles (71), `orin-refine1`, `orin-lin-refine2`, `orin-refine3` | screening and candidate profiles; the SDPA kernels they name are the 4070 Ti campaign's dev variants (same shapes: subgroup 32, MMA 16x16x16 fp16 -> fp32), unchanged | `gen_orin_sdpa.py`, `orin_refine.py` |
| `glsl/sarc_dev/sarc_sdpa_qk_coopmat_orin_pk` | 12 more packed-staging QK^T tiles (K = 64, K = 128); the shader is the sibling's, identical from `#version` on | `gen_orin_qk.py` |
| `glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_orin_bf` | **new**: 8da4w zpgtr with whole-texel weight staging (twin of the release body), 15 tiles, two forms (`bf`: two staging slices, `bf1`: one) | `gen_orin_bf.py` |
| `glsl/sarc_dev/sarc_dev_prof_orin_q4gsw`, `sarc_dev_prof_orin_dq8ca_bf` | phase-timing twins, measurement only | `gen_orin_prof.py` |
| `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` | one line: the SDPA pairing check reads the softmax kernel name by substring (see `STATUS.md`, "A test fix") | by hand |

## What cannot be reached from the dev zone

1. **Softmax name.** `sdpa_softmax_shader_name()` in `impl/sarc/SdpaCoopmat.cpp` returns a fixed name, so no dev
   variant can replace the release softmax (fp16 arithmetic, full zero tail). Smallest hook: let the dev override
   name the softmax (`tools/local-hook-orin-softmax.patch`, 9 lines, the sibling's hook 2). Measured through
   build `hook4` = `topic4` + that patch, applied to the archived source tree only and NOT committed.
2. Everything outside the linear and SDPA kernels: copy / view, elementwise, RMSNorm and the 8-bit activation
   quantisation are upstream kernels (15 to 25 % of the prefill after the candidates).

SDPA rows for the device are NOT in this list: `OrinSdpa.cpp` reaches the SARC SDPA path from the dev zone.

## Method

- Build host: the owner's workstation (x86); measurement device: `duck-naughty` (Jetson Orin Nano Super 8 GB,
  JetPack 7.2 / L4T R39.2.1, NVIDIA 595.78, 15 W mode, GPU devfreq 306 to 612 MHz, governor `nvhost_podgov`),
  reached over ssh. Nothing is compiled on the device; clocks, power mode and fan were never touched.
- Builds: `tools/build-orin.sh` = `git archive` tree of a commit with its 30 pinned submodules
  (`mktree.sh`; later trees are hard links to the parent's tree plus the changed files, verified by blob hash)
  cross-built with the recipe of the campaign that produced the Orin rows of `cells.csv`
  (`tools/jetson-cross/`, image `localhost/et-jetson-cross:jp7.2.1`, GCC 13.3, shaderc v2026.1, 8 jobs), under
  an exclusive `flock` on the workstation's build lock. One runner serves timed and traced runs (ETDump is
  linked, as in that campaign). Shipped SPIR-V: with this glslc 14 of the 53 shipped variants differ from the
  golden already on the pristine parent (none of them an Orin variant); `tools/shipped.py` therefore compares
  every build with the parent build: all 53 byte-identical in every build of this campaign.
- End to end: `tools/e2e5.sh` on the device (the kit `e2e.sh` protocol: fresh `llama_main` per run, `--warmup`,
  `prompt_2048.txt`, one new token, arms interleaved, cool between runs), 5 valid runs per arm and cell; clock,
  load, module power and temperature sampled every 0.1 s from sysfs. A run counts only with rc 0, 2048 prompt
  tokens, 0 generated tokens, no GPU job of another owner and a median clock of at least 593 MHz
  (`results/orin/clkmin.json`). One GPU job at a time under the gpu-lab lock. Memory and swap counters are
  recorded around every run.
- Gate: `tools/gate.sh` / `tools/gate_sdpa.sh` on the device, judged on the content of the result files by
  `tools/gate_check.py` (41 unit tests). The pristine parent itself fails the buffer production-diff cases and
  three upstream correctness cases on this device, so the checker compares the status of every correctness case
  and production-diff shape with the parent control instead of requiring a pass.
- Jobs on the device run detached with a status file (`tools/drun.sh`); results are mirrored to the artifact
  directory (`tools/pull.sh`) and the small evidence files copied here (`tools/collect.sh`).

## Baseline, A/A, roofs

Baseline re-measured (`s1-aa`, pristine parent against the topic build without environment):
1B 890.05 / 823.48, 3B 360.44 / 320.30, 8B 189.68 / 170.47 tok/s (4w / 8da4w), within 0.1 % of
`sarc-1.5-e2e-benchmark/results/cells.csv` in every cell. A/A geomean +0.01 %, largest cell 0.09 %, repeat
spread at most 0.36 %.

Fresh roofs (igpu-roofline `fast`, driver 595.78, run `raw/roof-2026-10-05-fast`, report in
`results/orin/roofline/2026-10-05-fast/`, clocks not pinned (612 MHz under load), every roof confirmed with 3
repeats within 1.4 %): matrix fp16 9.716 TFLOP/s, fp16 -> fp32 9.722, int8 19.482 TOP/s; fed from shared memory
9.511 / 8.771 / 17.818; DRAM read 62.1, write 58.0, copy 64.1 GB/s; texture2d read 20.1 GB/s from DRAM and
46 GB/s from cache.

## Where the time goes (parent)

Warm ETDump, ms per 2048-token prefill (`results/orin/sessions/s1-aa/trace/`):

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 667 | 871 | 1895 | 2577 | 4863 | 6054 |
| attention (QK^T + softmax + attn*V, all stock) | 1215 | 1215 | 2913 | 2912 | 4436 | 4435 |
| copy / view / other | 267 | 154 | 581 | 351 | 985 | 574 |
| elementwise | 101 | 101 | 190 | 190 | 363 | 377 |
| 8-bit quantize | - | 101 | - | 254 | - | 425 |
| total | 2285 | 2477 | 5666 | 6373 | 10773 | 11992 |

Attention is half of the 1B and 3B prefill and 37 to 41 % of the 8B one. Linear kernels in the model against
the fresh roofs: 4w 5.74 to 6.21 TFLOP/s = 59 to 64 % of the fp16 matrix roof (54 % for the fp32-accumulating
tile of 8B `w2`); 8da4w 4.39 to 4.85 TOP/s = 22.5 to 24.9 % of the int8 matrix roof.

DRAFT: the candidate sections follow when the gates have ended.
