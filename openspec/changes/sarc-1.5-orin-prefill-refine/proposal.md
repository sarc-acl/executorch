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

DRAFT: the rule's real-text evidence, candidate 2 and the final sections follow.
