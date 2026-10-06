# sarc-1.5-orin-prefill-refine

Dev zone only (2026-10-05). Branch `topic/orin-prefill-refine`, forked from `topic/4070ti-prefill-refine` at
`42e001462`. The parent of every comparison is commit `6a7cc8cc6` with no profile (the shipped NVIDIA rows).
Nothing is promoted and no upstream file is edited. One release-zone edit is committed, by owner decision: the
softmax-name hook (its own section below). Day-to-day state, incidents and the tool history are in `STATUS.md`.

## Why

Push the 2048-token Llama prefill on the Jetson Orin Nano (8 GB, 15 W mode, Vulkan driver NVIDIA 595.78) as far
as it goes, measured against the parent in the same session.

## Outcome

Stopped by the stop rule on 2026-10-06: candidates 3 and 4, two consecutive gated candidates, gained +0.82 %
and +0.41 % geomean over their parents. Four candidates are accepted; one release-zone hook is committed by
owner decision; nothing is promoted.

Final stack = `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`
on build `topic13` (`fd44f8011`, the last commit that changes code), against the pristine parent `6a7cc8cc6` in
the same session (`results/orin/sessions/s7-final/`; tok/s, median of 5 valid interleaved runs per arm, 60
timed runs all valid at 612 MHz, repeat spread at most 0.28 %):

| cell | parent | final | gain | `dev/1.5` (`cells.csv`) | gain over `dev/1.5` | next token (4 prompts) |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 890.82 | 1490.54 | +67.3 % | 890.82 | +67.3 % | SAME |
| 1B 8da4w | 824.14 | 1380.98 | +67.6 % | 822.82 | +67.8 % | SAME |
| 3B 4w | 360.44 | 629.38 | +74.6 % | 360.37 | +74.7 % | SAME |
| 3B 8da4w | 320.30 | 570.95 | +78.3 % | 320.30 | +78.3 % | SAME |
| 8B 4w | 189.77 | 295.61 | +55.8 % | 189.74 | +55.8 % | SAME |
| 8B 8da4w | 170.53 | 269.19 | +57.9 % | 170.43 | +57.9 % | SAME |

Geomean **+66.70 %** over the parent, +66.77 % over the original `dev/1.5` numbers. 8da4w is still slower than
4w end to end in every model, by 7 to 9 % (it was 7 to 11 %).

| # | what | gate | gain over its parent (geomean of six cells) | recorded as |
|---|---|---|---:|---|
| 1 | SDPA prefill kernels, release softmax | `s2-c1` | (+57.5 % measured) | **REJECTED**: one next-token item differs, reference-error rule not met in 1 of 12 cases |
| 1h | the same kernels + fp32 softmax without the zero tail (`4070ti_nzf`), through the hook | `s4-c1h` | +57.81 % (parent: pristine) | `ACCEPTED (reference-error rule, owner decision 2026-10-04)` |
| 2 | 8da4w linear: whole-texel weight staging (`orin-lin-refine2`) | `s3-c2` | +2.84 % (parent: pristine; 8da4w cells +3.8 / +6.9 / +6.7 %) | `GATE_ACCEPTED`, bit-identical to the shipped kernel |
| 3 | 4w linear tiles (`orin-refine5`) | `s5-c3` | **+0.82 %** (parent: 1h + 2) | `GATE_ACCEPTED`, bit-identical to the shipped kernels on the shapes served |
| 4 | softmax with subgroup reductions (`orin_g64`) | `s6-c4` | **+0.41 %** (parent: 1h + 2 + 3), inside the +-2 % band: not a gain by the protocol | `GATE_ACCEPTED`, no differing item; arithmetic change, reference error reported |

Where the gain is (warm ETDump of both arms of `s7-final`, ms per 2048-token prefill, parent -> final):

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---|---|---|---|---|---|
| QK^T | 581 -> 108 | 581 -> 108 | 1488 -> 198 | 1488 -> 198 | 2269 -> 301 | 2269 -> 301 |
| attn*V | 457 -> 61 | 457 -> 61 | 1194 -> 144 | 1195 -> 144 | 1818 -> 216 | 1818 -> 216 |
| softmax | 177 -> 132 | 177 -> 132 | 231 -> 173 | 231 -> 173 | 352 -> 264 | 352 -> 264 |
| linear GEMM | 666 -> 656 | 871 -> 779 | 1895 -> 1862 | 2578 -> 2171 | 4865 -> 4657 | 6057 -> 5305 |
| copy / view / other (upstream) | 266 -> 266 | 154 -> 154 | 580 -> 581 | 351 -> 351 | 985 -> 985 | 574 -> 574 |
| elementwise, 8-bit quantize, RMSNorm (upstream) | 119 -> 119 | 218 -> 216 | 236 -> 236 | 491 -> 490 | 435 -> 434 | 874 -> 873 |
| total dispatch | 2283 -> 1359 | 2476 -> 1467 | 5664 -> 3235 | 6373 -> 3568 | 10779 -> 6913 | 11997 -> 7589 |

Of the 924 ms saved on 1B 4w, 914 are attention (1215 -> 301) and 10 the 4w tile; of the 1009 ms on 1B 8da4w,
914 are attention and 92 the 8da4w linear kernel. On 8B 8da4w: 3658 of 4408 ms are attention, 752 the linear
kernel.

Against the fresh roofs (igpu-roofline `fast`, driver 595.78, run `raw/roof-2026-10-05-fast`; `tools/roof_util.py`
on the `gemm.csv` of `s7-final`, time-weighted over the model's linear layers):

| kernels | parent | final |
|---|---|---|
| 4w linear, share of the fp16 matrix roof (9.716 TFLOP/s) | 61.6 / 62.7 / 60.5 % (1B / 3B / 8B) | 62.6 / 63.8 / 63.2 % |
| 8da4w linear, share of the int8 matrix roof (19.482 TOP/s) | 23.5 / 23.0 / 24.2 % | 26.3 / 27.3 / 27.7 % |
| QK^T, 1B layer: 2.3 ms (134 MB written at the DRAM write roof) + 0.9 ms (8.6 GFLOP at the fp16 -> fp32 roof) | 36.3 ms | 6.73 ms = 48 % |
| attn*V, 1B layer: 2.2 ms (134 MB read) + 0.9 ms | 28.6 ms | 3.84 ms = 81 % |
| softmax, 1B layer: 2.2 ms read + 2.3 ms written | 11.0 ms | 8.25 ms = 55 % |

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
| `glsl/sarc_dev/sarc_linear_q4gsw_coopmat_orin` (sweep variants) | 4w: staging and grid variants of the shipped Orin tiles (column-major weight staging `bt`, texel-wise fetches `c`, other subgroup grids); candidate 3 uses `orin_t256x128k16g42s32bt` | `gen_orin_q4.py` |
| `glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_orin`, `..._orin_wg` | softmax variants of `4070ti_nzf`: 16 that keep a row after the first pass (negative), 24 with other reductions and workers per row (candidate 4 is `orin_g64`), 3 measurement-only pass twins (`orin_xp*`, wrong output by construction) | `gen_orin_softmax.py` |
| profiles `orin-lin-refine3`, `orin-refine4`, `orin-lin-refine5`, `orin-refine5` | the linear candidates alone and with the SDPA kernels; `orin-refine5` is everything | `orin_refine.py` |
| `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` | one line: the SDPA pairing check reads the softmax kernel name by substring (see `STATUS.md`, "A test fix") | by hand |

## Release-zone hook (owner decision 2026-10-05)

The owner answered the question raised in `STATUS.md` ("Decision needed from the owner") with option (a): one
release-zone edit, the softmax-name hook, is accepted for this campaign. It is commit `307abb2ed`, alone, so it
can be reviewed or reverted by itself (the dev-zone commit `4718f3e07` that uses the new field has to be
reverted with it, or the dev zone does not compile).

Form: the existing `Override` mechanism, not an environment variable in release code. `Override` gains one
field, `softmax_variant` (null by default); `sdpa_softmax_shader_name` appends it to the SARC softmax name.
With no override, or an override that names no variant (every release build, every configuration without the
dev zone's `ET_VK_SARC_SOFTMAX_VARIANT`), the function returns exactly what it returned before.

```diff
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
index d45f720bc..5a264e27e 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
@@ -202,7 +202,11 @@ std::string sdpa_softmax_shader_name(
   if (!device_has_active_rows(device_info(&graph), Op::kSdpaQk)) {
     return upstream_name;
   }
-  return "sarc_" + upstream_name;
+  const char* variant = get_override().softmax_variant;
+  if (variant == nullptr) {
+    return "sarc_" + upstream_name;
+  }
+  return "sarc_" + upstream_name + "_" + variant;
 }
 
 } // namespace sarc
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
index 93a9b8e96..166eba3e0 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
@@ -141,6 +141,9 @@ struct Override {
       const DeviceInfo& device,
       const ShapeInfo& shape,
       const std::optional<Choice>& table_choice) = nullptr;
+  // Suffix of a development softmax variant ("sarc_<softmax>_<suffix>"), or
+  // null: the release softmax.
+  const char* softmax_variant = nullptr;
 };
 void set_override(const Override& o);
 const Override& get_override();
```

- The variant is named from the dev zone: `impl/sarc_dev/Overrides.cpp`, block `orin softmax-variant`, reads
  `ET_VK_SARC_SOFTMAX_VARIANT` and prints `[sarc_dev] softmax variant: <suffix>`; `tools/gate_check.py` requires
  that banner, and the softmax kernel name in every SDPA correctness case, to match the candidate environment.
- Nothing else in the release zone changes: no table row, no shipped shader, no golden.
  `sarc/tools/check.sh --no-build` after the edit: `check.sh: PASS`; `test_sarc_select` with the release
  tables alone reads `PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)`, the same line
  as before the edit; shipped SPIR-V of the build that contains the hook: see "Builds" in `STATUS.md`
  (`tools/shipped.py`: all 53 shipped variants byte-identical to the parent build). `check.sh` does not flag the
  edit: its zone rule allows the release zone; the campaign's dev-zone-only rule is stricter, and this decision
  is the exception to it.
- Option (b) was not granted: candidate 1 (the release softmax) stays rejected.

## What cannot be reached from the dev zone

1. **Softmax name**: was not reachable; now the hook above (the sibling campaigns' "hook 2"). Before the
   decision it was measured through `tools/local-hook-orin-softmax.patch` (the environment-variable form,
   build `hook4`, never committed, kept in `tools/` for the record and no longer used).
2. Everything outside the linear and SDPA kernels: copy / view, elementwise, RMSNorm and the 8-bit activation
   quantisation are upstream kernels.

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

## SDPA prefill kernels: candidates 1 and 1h

The Orin has no SARC SDPA row; attention ran the upstream kernels (fp16 accumulation). `impl/sarc_dev/OrinSdpa.cpp`
registers two `kUnverified` base rows for `tegra orin` that match only under an `orin-*` profile (not
`orin-lin-*`), which enables the SARC SDPA path (spec constants, SARC softmax) from the dev zone. The kernels
are the 4070 Ti campaign's dev variants: the Orin exposes the same cooperative-matrix shapes (16x16x16 fp16 with
fp32 result, subgroup size 32 only; `results/orin/vk-caps.txt`).

Kernel per head_dim, chosen at kernel level (`results/orin/screens/sdpa-screen{1,2}.csv`: 59 + 21 profiles; ms
per layer at S = 2048):

| kernels | 1B QK^T / softmax / attn*V | 3B | 8B |
|---|---|---|---|
| stock (the parent) | 36.29 / 10.97 / 28.60 | 53.11 / 8.23 / 42.64 | 70.87 / 10.97 / 56.76 |
| `orin-refine1` (candidate 1) | 6.73 / 9.03 / 3.84 | 7.10 / 6.79 / 5.13 | 9.40 / 9.03 / 6.75 |
| + softmax `4070ti_nzf` (candidate 1h) | 6.73 / 8.75 / 3.84 | 7.10 / 6.55 / 5.14 | 9.39 / 8.74 / 6.75 |

QK^T: packed staging with K = 64 per chunk on a 128 x 64 tile (`4070ti_pk_t128x64k64g42s32nf`), all head dims.
attn*V: head_dim 64 the 64 x 64 tile, head_dim 128 `4070ti_ml_t64x128k32g42s32`. Direct feed, which won for
head_dim 64 on the 4070 Ti, loses on this device (10 to 20 ms for QK^T): feeding the matrix unit from DRAM is
slow here (fresh roofs: 2.8 TFLOP/s fed from DRAM against 8.8 fed from shared memory). Twelve further Orin
QK^T tiles (K = 64 on other tiles and grids, K = 128) were generated and screened: none is faster.

### Candidate 1 (`orin-refine1`, release softmax): +57.5 % measured, REJECTED

Gate `s2-c1`: 144 of 144 SDPA correctness cases with 0 mismatches and `pairing=ok`; unmodified `verify.sh` equal
to the parent control in every item except `1b 8da4w unaligned: default vs tiled output DIFFER`, the item the
4070 Ti's candidate 1 failed on. Under the reference-error rule the candidate is not larger than the parent in
rms and maximum error in 11 of 12 cases; on the 3B production case its maximum error is 1.570e-3 against the
parent's 1.408e-3. Not met; rejected. The owner confirmed this on 2026-10-05 (option (b) not granted).
Evidence session (verdict unchanged): 1B 1466.0 / 1290.5, 3B 620.2 / 510.2, 8B 286.0 / 244.3 tok/s, geomean
+57.5 %, next token parent vs candidate SAME in 24 of 24 rows.

### Candidate 1h (`orin-refine1` + `ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf`): +57.8 %, GATE_ACCEPTED

The softmax reduces each row in fp32 (maximum, exp, sum, division), rounds once on the store and does not write
the masked tail. It is the 4070 Ti campaign's dev shader, selected through the release-zone hook above.
Build `topic6` = `4718f3e07`, gate `s4-c1h` against the pristine parent:

- Error against the fp32 CPU reference (`results/orin/sdpa-error2/summary.txt`), production cases, rms / maximum:
  1B 2.10e-5 / 0.914e-3 (parent 8.55e-5 / 1.713e-3), 3B 2.07e-5 / 0.783e-3 (8.69e-5 / 1.408e-3), 8B 2.06e-5 /
  0.891e-3 (8.70e-5 / 1.587e-3). Not larger than the parent in 12 of 12 cases, both measures.
- 144 of 144 SDPA correctness cases: 0 mismatches, `pairing=ok`, softmax kernel `..._4070ti_nzf`.
- Unmodified `verify.sh`: `gate_check.py verify` ACCEPT with 0 findings; all four default-vs-tiled next-token
  items SAME.
- Session (tok/s, median of 5 valid interleaved runs per arm):

| cell | parent | candidate 1h | gain | `dev/1.5` (`cells.csv`) | ETDump dispatch total, ms |
|---|---:|---:|---:|---:|---|
| 1B 4w | 890.82 | 1471.26 | +65.2 % | 890.82 | 2283 -> 1378 |
| 1B 8da4w | 822.82 | 1292.93 | +57.1 % | 822.82 | 2475 -> 1570 |
| 3B 4w | 360.50 | 621.36 | +72.4 % | 360.37 | 5662 -> 3278 |
| 3B 8da4w | 320.45 | 511.23 | +59.5 % | 320.30 | 6372 -> 3986 |
| 8B 4w | 189.74 | 286.23 | +50.9 % | 189.74 | 10776 -> 7137 |
| 8B 8da4w | 170.45 | 244.57 | +43.5 % | 170.43 | 11995 -> 8363 |

  Geomean +57.81 %; next token parent vs candidate SAME in 24 of 24 rows; 60 timed runs, all valid.
- The gain is attention alone (ETDump, ms): 1B 1215 -> 309, 3B 2912 -> 526, 8B 4437 -> 798; every other family
  is unchanged within 1 ms.

- Reference-error rule, as written (`results/orin/probe/refine1-nzf/`): criterion 1 met in 12 of 12 cases; the
  41-prompt real-text comparison of the four arms shows, per cell, top-1 differences candidate vs parent of 0 /
  5 / 0 / 1 / 0 / 4 of 41 (the parent's own two arms: 0 / 2 / 0 / 1 / 0 / 1), mean KL at most 0.076 nat (limit
  0.5), perplexity ratio 0.996 to 1.078; no gross divergence; no differing next-token item in the gate. Recorded
  as `ACCEPTED (reference-error rule, owner decision 2026-10-04)`. The table is in `STATUS.md`.

## 8da4w linear: candidate 2 (whole-texel weight staging), +2.84 % alone, GATE_ACCEPTED

Phase timing of the shipped `zpgtr_t128x128k64g44s32mk32ra` on this device (shader clock, share of a wave;
`results/orin/phases/prof1-8da4w.csv`): weight and activation fetch 38 to 47 %, MMA 24 to 31 %, barrier 10 to
15 %. The shipped kernel fetches every packed-weight texel four times per chunk and keeps one 32-bit word of
it each time, and a texture2d read is the slowest path the roofs show (20 GB/s from DRAM against 62 for a
buffer). `glsl/sarc_dev/sarc_linear_dq8ca_coopmat_zpgtr_orin_bf` is a twin of the release body in which a
staging slot is a whole texel (one fetch, eight shared-memory words): same values, same shared-memory layout,
same MMA order.

- Kernel level, geomean over the 12 model shapes against the shipped kernel (`results/orin/screens/
  screen{1,2,3}-8da4w.txt`, 30 tiles in all): whole texel on the shipped 512-thread tile 1.019 (every second
  thread has no texel), on a 256-thread 128 x 128 tile **1.158** (`g24`; `g42` 1.154); one staging slice with a
  second barrier per chunk 0.92 to 0.99; smaller tiles 0.68 to 1.02; the 4070 Ti sweep tiles 0.69 to 0.87. What
  matters is that every thread makes one fetch per chunk. Phase timing of the chosen tile: fetch 15 to 24 %,
  MMA 48 to 59 %.
- Bit-identical to the shipped kernel on all 12 shapes (`screens/bit-bf.txt`); production-diff with non-zero
  zero points ALL PASSED on the three models.
- Gate `s3-c2` (`ET_VK_SARC_DEV_PROFILE=orin-lin-refine2` against the pristine parent, build `topic6`):
  `verify.sh` equal to the parent control, 0 findings; 1B 8da4w 823.81 -> 854.76 (+3.76 %), 3B 320.45 -> 342.59
  (+6.91 %), 8B 170.51 -> 181.88 (+6.67 %), the 4w cells within 0.02 %; geomean +2.84 %; next token SAME in 24
  of 24 rows. ETDump: linear GEMM of the 8da4w cells 870 -> 778, 2577 -> 2171, 6054 -> 5306 ms.
- On top of candidate 1h the same milliseconds weigh more: the 8da4w cells of the stack 1h + 2 read 1374.5 /
  569.1 / 268.6 tok/s against 1292.9 / 511.2 / 244.6 for 1h alone (+6.3 / +11.3 / +9.8 %; two sessions).

## 4w linear: candidate 3, +0.82 % over its parent, GATE_ACCEPTED

Phase timing of the shipped Orin 4w tiles (`results/orin/phases/prof2-4w.csv`): MMA 45 to 49 %, weight
dequantisation into shared memory 18 to 20 %, fetch 17 to 18 %, barrier 10 to 11 % (the fp32 tile of 8B `w2`:
MMA 30 to 37 %, store 26 to 38 %). Two screens at kernel level, 37 tiles in all (`screens/screen{1,2}-4w.txt`):
no existing subgroup-32 dev tile beats the shipped rows (best 0.94x), except on the fp32-accumulating shape
(K = 14336), where texel-wise weight staging is 1.11x; of the staging and grid variants of the shipped tiles
themselves, column-major weight staging on a 4 x 2 grid is 1.3 to 1.8 % faster on every K <= 8192 shape, and
nothing else is. Candidate 3 = profile `orin-refine5` = those two tiles, each bit-identical to the shipped
kernel on the shapes it serves (`screens/bit-q4.txt`, `bit-q4b.txt`).

Gate `s5-c3` (build `topic13`, parent arm = candidates 1h + 2, candidate arm = the same + the 4w tiles):
`verify.sh` equal to the parent control, 0 findings; 1B 4w 1472.32 -> 1484.06 (+0.80 %), 3B 621.55 -> 627.84
(+1.01 %), 8B 286.43 -> 294.97 (+2.98 %), the 8da4w cells within 0.13 %; geomean **+0.82 %**; next token SAME
in 24 of 24 rows. ETDump: linear GEMM of the 4w cells 667 -> 655, 1895 -> 1862, 4862 -> 4654 ms.

## Softmax after candidate 1h: one negative family, candidate 4

After candidate 1h the softmax is the largest attention kernel (8.74 ms of the 19.3 ms a 1B layer's attention
takes). Two hypotheses, both tested at kernel level (`tools/gen_orin_softmax.py`):

1. *The three passes re-read the row from DRAM.* Negative. Keeping the row after the first pass (local array
   or shared memory, 16 variants; `screens/sdpa-screen4.csv`) is 1.12x to 4.7x slower. The re-reads are cache
   hits; and a workgroup that declares 8 / 16 / 32 KB of shared memory runs 1.9x / 2.6x / 4.6x longer.
2. *The 14 barriers per row (two 64-worker tree reductions) and the split of a row over workers.* Partly.
   Fewer workers per row is slower in proportion (one thread per row: 18x), more than 128 is slower again;
   reducing inside each subgroup with `subgroupMax` / `subgroupAdd` and one barrier is the fastest form:
   `orin_g64` 8.25 / 6.18 / 8.25 ms against 8.74 / 6.53 / 8.74 on the primary (`screens/sdpa-screen7.csv`),
   -5.6 %. Pass timing (measurement twins, second device): workgroup launch 1.0 ms, the first read and the
   maximum 4.1, exp and sum 1.6, exp, divide and store 1.5: half of the softmax is the first read of the row.

`orin_g64` sums a row's exponentials in another order (fp32, rounded once on the store), so it is an arithmetic
change. Against the fp32 CPU reference its rms and maximum error equal `4070ti_nzf`'s to four digits in all 12
cases and are 4 times (rms) and 2 times (maximum) below the parent's (`results/orin/sdpa-error5/summary.txt`);
0.3 to 0.4 % of the fp16 output elements differ from `4070ti_nzf`'s, by one fp16 step at most.

Gate `s6-c4` (build `topic13`; parent arm = candidates 1h + 2 + 3, candidate arm = the same with
`ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`): 144 of 144 SDPA correctness cases with 0 mismatches, `pairing=ok` and the
`orin_g64` kernel; `verify.sh` equal to the parent control, 0 findings, all four default-vs-tiled items SAME;
session 1B 1482.98 -> 1491.62 / 1373.57 -> 1385.66, 3B 628.03 -> 629.77 / 569.05 -> 570.79, 8B 295.06 -> 295.78
/ 268.66 -> 269.15 (4w / 8da4w): geomean **+0.41 %**, every cell inside the +-2 % band, next token SAME in 24 of
24 rows. ETDump: softmax 140 -> 132, 183 -> 173, 280 -> 264 ms. By the protocol this is noise, not a gain; it is
in the final stack because the gate accepted it, and the stack without it is that session's parent arm.

REALTEXT_PLACEHOLDER

## The second Orin (`duck-stable`)

Offered by the owner for screening. Agreement batch of 26 linear configurations: rank correlation 0.997, 25
ratios within 0.975 to 1.002, one (a 1024-thread tile, never selectable) at 0.80, reproducibly (3 rounds per
device: 0.81). By the threshold fixed beforehand that is DISAGREE and I stopped; the owner then decided to use
the device for screening regardless. It screened two softmax batches. Its ranking of the two best softmax
variants (`g128` before `g64`, 10 to 11 % faster than `4070ti_nzf`) was not confirmed on the primary (`g64`
before `g128`, 4.4 to 5.6 %), so, as that decision says, it was not used again. The two devices agree within
0.3 % on matrix-bound kernels and differ by 1 to 7 % on memory-bound ones (the same softmax: 8.16 against
8.74 ms). No number in this document comes from it except where it says so.

## Reproducible from the committed branch alone

- Every accepted candidate was gated on a build of a commit of this branch, cross-built from a `git archive`
  tree (`build/<tag>.src.txt`): 1h on `topic6` = `4718f3e07`, which already contains the committed hook
  `307abb2ed`; 2 on `topic6`; 3, 4 and the final session on `topic13` = `fd44f8011`. No commit after
  `fd44f8011` changes anything outside this directory. No local patch is in any of these builds.
- The only build with a local patch was `hook4` (the environment-variable form of the hook, before the owner's
  decision). It was used for one kernel-level softmax screen (`screens/sdpa-screen3.csv`): `4070ti_nzf` 8.75 /
  6.55 / 8.74 ms per layer; the committed builds read 8.74 / 6.54 / 8.74 (`topic10`) and 8.74 / 6.53 / 8.74
  (`topic13`). No end-to-end number was measured through the patch.
- The kernels of candidates 1h and 2 are byte-identical SPIR-V in `hook4`, `topic6` and `topic13`
  (`build/<tag>.spv.sha256`: the QK^T tile, both attn*V tiles, the `4070ti_nzf` softmax, the `orin_bf` tile), and
  the gates of `s5-c3` and `s6-c4` ran the whole stack on `topic13` with the same kernel names.
- With nothing selected the branch dispatches what the parent dispatches: `s8-noenv` = unmodified `verify.sh` on
  `topic13` without any environment: `gate_check.py verify` against the parent control ACCEPT with 0 findings,
  and `verify.out` equal to the parent control's line by line once the rates are removed (0 differing lines,
  `sessions/s8-noenv/verify-lines.txt`). `test_sarc_select` on the release tables: `PASS (1240 checks, 31 rows,
  0 candidates, dev zone absent, unverified off)`, the line of the parent. Shipped SPIR-V: all 53 shipped
  variants byte-identical between the parent build and every build of this campaign (`tools/shipped.py`).

## What limits further progress

1. **Linear layers are now 48 % (1B 4w) to 70 % (8B 8da4w) of the prefill**, and neither scheme is bound by
   the tile: 37 4w tiles and 30 8da4w tiles were screened. 4w sits at 63 % of the fp16 matrix roof with the
   MMA at 45 to 49 % of a wave (weight dequantisation into shared memory 18 to 20 %, fetch 17 %, barriers
   10 %). 8da4w is at 27 % of the int8 roof; after candidate 2 the MMA is half of a wave and the rest is
   fetch (15 to 24 %), barrier (10 %) and store (7 %). On this device int8 has twice the fp16 matrix rate
   and 8da4w is still the slower scheme end to end: its kernel keeps half of each wave for something other
   than multiplying, and 8da4w pays 98 to 425 ms of upstream 8-bit quantisation on top.
2. **Upstream kernels that the dev zone cannot reach** (copy / view, elementwise, RMSNorm, 8-bit quantise) are
   385 of 1359 ms on 1B 4w (28 %), 370 of 1467 on 1B 8da4w, 1419 of 6913 on 8B 4w (21 %). They did not move.
3. **Attention is bound by memory traffic, not by the matrix unit.** What is left of it (301 of 1359 ms on 1B)
   writes the S x S attention weights to DRAM (QK^T), reads and rewrites them (softmax) and reads them again
   (attn*V). The softmax measurements say where: half of its time is the first read of a row. A fused
   attention kernel that never stores the S x S matrix is the remaining structural step; it needs its own
   node, an entry point outside the dev zone (the 780M campaign's fused attention node), which this campaign
   was not given and did not build.
4. **Shared memory is expensive on this device.** A 64-thread workgroup that declares 8 / 16 / 32 KB of shared
   memory ran 1.9x / 2.6x / 4.6x longer in the softmax screen, and the only 1024-thread linear tile was the one
   configuration on which two identical Orins disagreed. The linear kernels stage whole tiles in shared memory
   (the device allows 48 KB per workgroup). I did not
   find a linear tile that wins by declaring less (smaller tiles lose more than they gain), but a kernel
   designed around that limit, not around the desktop cards' tiles, has not been tried.

## Negative results (kept with their numbers)

- QK^T: direct feed of the matrix unit from DRAM loses on this device (10 to 20 ms against 6.7 with packed
  staging; it won on the 4070 Ti); 12 further packed tiles (K = 64 on other grids, K = 128): none faster.
- fp16-accumulating attention variants: no gain, not pursued.
- 8da4w: one staging slice with a second barrier per chunk (K = 128): 0.92 to 0.99x; smaller tiles 0.68 to
  1.02x; the 4070 Ti sweep tiles 0.69 to 0.87x.
- 4w: every existing subgroup-32 dev tile (25) is slower than the shipped Orin rows except on the fp32 shape;
  K = 32 tiles 8 to 18 % slower; 2 x 4 and 4 x 4 subgroup grids 4 to 37 % slower.
- Softmax that reads its row once: 1.12x to 4.7x slower (16 variants). Fewer than 64 workers per row: slower in
  proportion. One thread walking the reduction tree: 1.76x slower.
- The second Orin as a screening device: agreement on matrix-bound kernels (rank correlation 0.997), not on
  memory-bound ones; its softmax ranking was not confirmed on the primary.

## Not done

- No fused attention kernel (item 3 above).
- No Orin-specific linear kernel beyond staging changes of the existing bodies (item 4 above).
- Decode was not optimized; the candidates leave it within the +-2 % band (decode A/B of candidates 1 and 2,
  `results/orin/decode/`).
- Only the 2048-token prefill shapes of the three models are covered; the Orin rows carry shape predicates
  and the profiles never extend them (a shape the table does not serve keeps the stock kernel).
